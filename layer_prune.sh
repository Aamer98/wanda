
model="llama-2"
prune="mhsa"
data="openwebtext"


CUDA_VISIBLE_DEVICES=0 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.0 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=0 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.05 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.15 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.25 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &



CUDA_VISIBLE_DEVICES=0 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.30 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.35 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.40 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.45 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.50 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.55 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.60 --dataset $data --model_name $model --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme $prune --wandb_project 'layer_influence' &


# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &



# CUDA_VISIBLE_DEVICES=1 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "openwebtext" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mhsa" --wandb_project 'layer_influence' &



# CUDA_VISIBLE_DEVICES=1 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "openwebtext" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "wikitext-2" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=3 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "cc_news" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=4 python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.1 --dataset "bookcorpus" --model_name "llama" --model_size 7b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "mlp" --wandb_project 'layer_influence' &





# CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &




# CUDA_VISIBLE_DEVICES=5 python layer_influence/evaluate_layer_influence.py --dataset "c4" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=6 python layer_influence/evaluate_layer_influence.py --dataset "slimpajama" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=2 python layer_influence/evaluate_layer_influence.py --dataset "pg19" --model_name "qwen2" --model_size 0.5b --batch_size 1 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

