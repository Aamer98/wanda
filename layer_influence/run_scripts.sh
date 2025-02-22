
# CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --dataset "openwebtext" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence'

CUDA_VISIBLE_DEVICES=1 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "wikitext-2" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=2 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "pg19" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=3 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "cc_news" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=4 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "bookcorpus" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=5 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "c4" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

CUDA_VISIBLE_DEVICES=6 python /home/aamer/repos/wanda/layer_influence/evaluate_layer_influence.py --dataset "slimpajama" --model_name "qwen2" --model_size 0.5b --batch_size 8 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &

# CUDA_VISIBLE_DEVICES=7 python layer_influence/evaluate_layer_influence.py --dataset "wikitext-2" --model_name "qwen2" --model_size 1.5b --batch_size 4 --sequence_length 2048 --subsample_size 250000 --pruning_scheme "both" --wandb_project 'layer_influence' &