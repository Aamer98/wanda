import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

from llama import Llama
from llama.model import ModelArgs, Transformer
from calflops import calculate_flops



print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

llama_seqlen = 2048

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        torch_dtype="auto",
        # cache_dir=cache_dir,
        # low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = min(llama_seqlen, model.config.max_position_embeddings)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "random", "wanda", "sparsegpt",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default="/home/aamer/repos/wanda/weights/llm/pruned", help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")

    # slimming
    parser.add_argument('--slimming', action="store_true", help="whether to use slimming")

    args = parser.parse_args()
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    if args.slimming:
        ckpt_dir = '/home/as26840@ens.ad.etsmtl.ca/.llama/checkpoints/Llama-2-7b'
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=512,
            max_batch_size=1,
            **params,
        )
        model_args.slimming = args.slimming
        model = Transformer(model_args)

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, truncation=True, max_length=model.seqlen)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    # Compute FLOPs
    flops, macs, params = calculate_flops(model=model.to(device),
                                      input_shape=(1,512),
                                      transformer_tokenizer=tokenizer)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_Method{args.prune_method}_{args.model.split('/')[-1]}_sparse{args.sparsity_ratio}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test\tFLOPs\tMACs", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{flops}\t{macs}", file=f, flush=True)




    # print("Model: %s Method: %s FLOPs:%s   MACs:%s   Params:%s \n" %(args.model, args.prune_method, flops, macs, params))


    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model_save_path = os.path.join(args.save_model, f"{args.prune_method}_{args.model.split('/')[-1]}_sparse{args.sparsity_ratio}")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()