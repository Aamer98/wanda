
CUDA_VISIBLE_DEVICES=5 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "openwebtext" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=6 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=1 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &



# CUDA_VISIBLE_DEVICES=5 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "openwebtext" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=6 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=1 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &



# CUDA_VISIBLE_DEVICES=1 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "openwebtext" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &





# CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=5 python layer_influence/evaluate_layer_influence.py --dataset "c4" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=6 python layer_influence/evaluate_layer_influence.py --dataset "slimpajama" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --dataset "pg19" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

