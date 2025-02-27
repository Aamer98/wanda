model="llama-3.2"
msize="1b"
prune="both"
data="openwebtext"
cuda_device=1


CUDA_VISIBLE_DEVICES=$cuda_device python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size $msize --batch_size 1 --sequence_length 2048 --subsample_size 10000 --pruning_scheme $prune --wandb_project 'layer_influence' 

CUDA_VISIBLE_DEVICES=$cuda_device python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size $msize --batch_size 1 --sequence_length 2048 --subsample_size 20000 --pruning_scheme $prune --wandb_project 'layer_influence' 

CUDA_VISIBLE_DEVICES=$cuda_device python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size $msize --batch_size 1 --sequence_length 2048 --subsample_size 50000 --pruning_scheme $prune --wandb_project 'layer_influence' 

CUDA_VISIBLE_DEVICES=$cuda_device python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size $msize --batch_size 1 --sequence_length 2048 --subsample_size 75000 --pruning_scheme $prune --wandb_project 'layer_influence' 

CUDA_VISIBLE_DEVICES=$cuda_device python layer_influence/evaluate_layer_influence.py --sparsity_ratio 0.20 --dataset $data --model_name $model --model_size $msize --batch_size 1 --sequence_length 2048 --subsample_size 100000 --pruning_scheme $prune --wandb_project 'layer_influence' 

