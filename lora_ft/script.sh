CUDA_VISIBLE_DEVICES=1 python finetune_lm.py \
    --model_name_or_path //home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-3B_sparse0.25 \
    --config_name "Qwen/Qwen2.5-3B" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &

# CUDA_VISIBLE_DEVICES=3 python finetune_lm.py \
#     --model_name_or_path /home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-3B_sparse0.5 \
#     --config_name "Qwen/Qwen2.5-3B" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 1024 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &


# CUDA_VISIBLE_DEVICES=4 python finetune_lm.py \
#     --model_name_or_path //home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-3B_sparse0.25 \
#     --config_name "Qwen/Qwen2.5-3B" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 1024 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &

# CUDA_VISIBLE_DEVICES=5 python finetune_lm.py \
#     --model_name_or_path /home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-3B_sparse0.75 \
#     --config_name "Qwen/Qwen2.5-3B" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 1024 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &


# CUDA_VISIBLE_DEVICES=6 python finetune_lm.py \
#     --model_name_or_path /home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-1.5B_sparse0.25 \
#     --config_name "Qwen/Qwen2.5-1.5B" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 1024 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &


# CUDA_VISIBLE_DEVICES=7 python finetune_lm.py \
#     --model_name_or_path /home/aamer/repos/wanda/weights/llm/pruned/wanda_Qwen2.5-3B_sparse0.75 \
#     --config_name "Qwen/Qwen2.5-3B" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 1024 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir /home/aamer/repos/wanda/weights/llm/finetuned &



# CUDA_VISIBLE_DEVICES=1 python evaluate_ppl.py \
#     --model /home/aamer/repos/wanda/weights/llm/pruned/Methodwanda_Qwen2.5-0.5B_sparse0.0 \
#     --lora_weights /home/aamer/repos/wanda/weights/llm/finetuned