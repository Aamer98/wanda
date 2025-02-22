#!/bin/python

import os
import time
import json
import random
import argparse
import torch
import numpy as np
import wandb
import csv
from tqdm import tqdm
from typing import Tuple

from transformers import AutoTokenizer, AutoConfig

from llama_layer_influence import LlamaForCausalLM
from qwen_layer_influence import Qwen2ForCausalLM
# from mistral_layer_influence import MistralForCausalLM

import sys
sys.path.append('.')
from dataset import NLPDataset, get_dataloader
from train_utils import get_num_model_params
from dist_utils import init_distributed_env, is_main_proc, wait_for_other_procs, reduce_tensor
from block_influence import BlockInfluenceEstimator
from evals.dist_mmlu import MMLUDataset, evaluate_mmlu
from utils import layer_importance_plot
from lib.eval import eval_ppl, eval_zero_shot
from calflops import calculate_flops


def load_model(args, only_tokenizer=False, pretrained=False):
    """
    Load the model and tokenizer based on the specified arguments.
    """
    # Model selection based on user input
    if args.model_name == "llama-2":
        model_name = f"meta-llama/Llama-2-{args.model_size.lower()}-chat-hf" if args.use_instruct_model else f"meta-llama/Llama-2-{args.model_size.lower()}-hf"
    elif args.model_name == "mistral":
        model_name = f"mistralai/Mistral-{args.model_size.upper()}-Instruct-v0.2" if args.use_instruct_model else f"mistralai/Mistral-{args.model_size.upper()}-v0.1"
    elif args.model_name == "qwen2":
        model_name = f"Qwen/Qwen2.5-{args.model_size.upper()}-chat-hf" if args.use_instruct_model else f"Qwen/Qwen2.5-{args.model_size.upper()}"
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")

    print("Loading model:", model_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if only_tokenizer:
        return tokenizer

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name)
    print("Model config:", config)

    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    print("Model precision:", kwargs["torch_dtype"])

    # Load the model itself
    if args.model_name == "llama-2":
        model = LlamaForCausalLM(config).to(kwargs["torch_dtype"]) if not pretrained else LlamaForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_name == "qwen2":
        model = Qwen2ForCausalLM(config).to(kwargs["torch_dtype"]) if not pretrained else Qwen2ForCausalLM.from_pretrained(model_name)
    elif args.model_name == "mistral":
        raise NotImplementedError("Mistral model loading is not implemented in this script.")  # Uncomment and complete if necessary
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")

    model.seqlen = min(2048, model.config.max_position_embeddings)
    
    return model, tokenizer


def compute_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log probabilities and perplexity given logits and target ids.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = log_probs.sum(dim=1).cpu().float().numpy()

    sequence_length = target_ids.size(-1)
    sequence_perplexity = np.exp(-sequence_log_prob / sequence_length)

    return sequence_perplexity, sequence_log_prob


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device, split_name: str):
    """
    Evaluate the model on the given dataset.
    """
    model.eval()
    avg_sequence_perplexity, avg_loss, num_ex = 0., 0., 0

    for batch in tqdm(eval_loader):
        tokenized_input = batch["input_ids"].to(device)

        # Forward pass through the model
        outputs = model(tokenized_input, labels=tokenized_input)

        lm_logits = outputs.logits[:, :-1, :]
        target_ids = tokenized_input[:, 1:]
        perplexity, _ = compute_log_probs(lm_logits, target_ids)

        avg_sequence_perplexity += float(perplexity.sum())
        avg_loss += float(outputs.loss)
        num_ex += len(tokenized_input)
        
    # Aggregate stats across all processes
    avg_sequence_perplexity = float(reduce_tensor(torch.tensor(avg_sequence_perplexity).to(device)))
    avg_loss = float(reduce_tensor(torch.tensor(avg_loss).to(device)))
    num_ex = int(reduce_tensor(torch.tensor(num_ex).to(device)))

    avg_sequence_perplexity /= num_ex
    avg_loss /= num_ex

    output_dict = {"split": split_name, "num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}
    print(json.dumps(output_dict))

    if split_name and wandb.run:
        wandb.log({f"eval_{split_name}": {"num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}})

    return avg_loss, avg_sequence_perplexity


def main(args):
    """
    Main function to run the evaluation and layer pruning process.
    """
    init_distributed_env(args)

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        print(f"Setting process seed: {args.seed}")

    # Dataset and directory setup
    dataset_dir = f"{args.dataset}_model_{args.model_name}_seq_len_{args.sequence_length}_subsample_{args.subsample_size}_comb_docs"
    args.dataset_output_dir = os.path.join("datasets", dataset_dir)

    # Generate a unique WandB run name
    args.wandb_run_name = f"{dataset_dir}_layer_pruning_{args.pruning_scheme}"

    # Initialize Weights & Biases
    if args.wandb_project and is_main_proc():
        print("Initializing wandb...")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, resume=False)
    
    # Process the dataset if necessary
    if is_main_proc() and not NLPDataset.is_dataset_processed(args.dataset_output_dir):
        tokenizer = load_model(args, only_tokenizer=True)
        dataset = NLPDataset(args.dataset, tokenizer, max_length=args.sequence_length,
                             combine_documents=True, subsample_size=args.subsample_size)
        dataset.save_datasets(args.dataset_output_dir)

    # Wait for other processes to sync
    wait_for_other_procs()

    # Load dataset and model
    dataset = NLPDataset.load_dataset(args.dataset_output_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    model, tokenizer = load_model(args, pretrained=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers)
    eval_loader = get_dataloader(test_dataset, args.test_batch_size, args.num_workers)

    # Load MMLU dataset for evaluation
    mmlu_dataset = MMLUDataset(tokenizer, args.model_name)
    mmlu_loader = get_dataloader(mmlu_dataset, 1, args.num_workers)

    # Pruning and influence computation
    total_layers = model.get_num_model_layers() * 2  # MHSA + MLP
    print(f"Total model layers: {total_layers}")

    # Decide whether to use block influences or compute them
    if args.use_block_influence:
        print("Using precomputed block influences for layer pruning...")
        # Block influence computation here (use predefined influences as in the original code)
        layer_influence_list = [("block_cosine", cosine_layer_influences)]
    else:
        print("Estimating layer influences...")
        block_influence_estimator = BlockInfluenceEstimator(total_layers, device)
        model.add_layer_influence_estimator(block_influence_estimator)

        # Evaluate to get block influences
        evaluate_model(model, train_loader, device, split_name=None)
        final_layer_influences = block_influence_estimator.get_block_influences()
        model.add_layer_influence_estimator(None)

        # Log and store layer influences
        cosine_layer_influences = final_layer_influences
        print("Final cosine layer influences:", cosine_layer_influences)

        if wandb.run is not None:
            for i, influence in enumerate(cosine_layer_influences):
                wandb.log({f"layer_{i}_cosine_influence": influence})

        layer_influence_list = [("cosine", cosine_layer_influences)]


    # Layer pruning according to layer importance
    layer_importances = [(cosine_distance, idx) for idx, cosine_distance in enumerate(cosine_layer_influences)]
    layer_importances.sort(key=lambda x: x[0], reverse=True)
    layers_to_keep = layer_importances[:int(total_layers * (1 - args.sparsity_ratio))]
    layers_to_keep = [idx for _, idx in layers_to_keep]
    layers_to_prune = layer_importances[int(total_layers * (1 - args.sparsity_ratio)):]
    layers_to_prune = [idx for _, idx in layers_to_prune]
    print(f"Layers to keep: {layers_to_keep}")

    model.select_layers(layers_to_keep)  # use the selected layers


    # Wikitext perplexity evaluation
    wikitext_perplexity = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {wikitext_perplexity}")


    # MMLU Eval
    _, _, weighted_acc = evaluate_mmlu(model, tokenizer, mmlu_loader, device, f"cosine_influence_sparsity_{args.sparsity_ratio}")
    _, avg_perplexity = evaluate_model(model, eval_loader, device, f"cosine_influence_sparsity_{args.sparsity_ratio}")

    # Compute FLOPs
    flops, macs, params = calculate_flops(model=model.to(device), input_shape=(1, 512), transformer_tokenizer=tokenizer)

    # Save results
    save_path = os.path.join(args.save_path, f'{args.pruning_scheme}_{args.model_name}{args.model_size}_sparse{args.sparsity_ratio}_{args.dataset}_samplesize{args.subsample_size}')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'layer_importance.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Layer', 'Layer Importance (cosine distance)'])
        writer.writeheader()
        for i, avg_dist in enumerate(cosine_layer_influences):
            writer.writerow({'Layer': i + 1, 'Layer Importance (cosine distance)': avg_dist})

    # Save evaluation results
    with open(os.path.join(save_path, "test_results.txt"), "w") as f:
        print(f"method\tactual_sparsity\twiki_perplexity\t{args.dataset}_ppl\tMMLU_acc\tFLOPs\tMACs", file=f)
        print(f"{args.pruning_scheme}\t{args.sparsity_ratio:.4f}\t{wikitext_perplexity:.4f}\t{avg_perplexity}\t{weighted_acc}\t{flops}\t{macs}", file=f)
        print(f"Layers to keep: {layers_to_keep}")
        print(f"Layers to remove: {layers_to_prune}")

    # Plot and save the layer importance
    layer_indexes = []
    for i in range(len(cosine_layer_influences)):
        layer_type = "mhsa" if i % 2 == 0 else "mlp"  # Odd index = mhsa, Even index = mlp
        layer_indexes.append(f"{layer_type}{(i // 2) + 1}")  # Index layers by 1-based ID
    
    layer_importance_plot(cosine_layer_influences, layer_indexes, save_path)

    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()

    print("Script execution completed!")


if __name__ == "__main__":
    supported_datasets = ['pg19', 'cc_news', 'wikitext-2', 'bookcorpus', 'c4', 'openwebtext', 'slimpajama']

    # Argument parser setup
    parser = argparse.ArgumentParser(description='LLM Layer Influence Evaluator')
    parser.add_argument('-d', '--dataset', default='wikitext-2', choices=supported_datasets)
    parser.add_argument('-m', '--model_name', default='llama-2', choices=['llama-2', 'mistral', 'qwen2'])
    parser.add_argument('-s', '--model_size', default='7b', choices=['7b', '3b', '1.5b', '0.5b'])
    parser.add_argument('--use_instruct_model', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--sequence_length', type=int, default=1024)
    parser.add_argument('--subsample_size', type=int, default=1000000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--pruning_scheme', default='both', choices=['mhsa', 'mlp', 'both'])
    parser.add_argument('--use_block_influence', action='store_true', default=False)
    parser.add_argument('--sparsity_ratio', type=float, default=0.25)
    parser.add_argument('--save_path', type=str, default='/home/aamer/repos/llm_depth_pruning/logs')

    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    main(args)