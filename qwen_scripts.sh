# CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-0.5B --prune_method magnitude --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-1.5B --prune_method magnitude --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=4 python main.py --model Qwen/Qwen2.5-3B --prune_method magnitude --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=4 python main.py --model Qwen/Qwen2.5-7B --prune_method magnitude --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &


# CUDA_VISIBLE_DEVICES=5 python main.py --model Qwen/Qwen2.5-0.5B --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=6 python main.py --model Qwen/Qwen2.5-1.5B --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=7 python main.py --model Qwen/Qwen2.5-3B --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=5 python main.py --model Qwen/Qwen2.5-7B --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &


# CUDA_VISIBLE_DEVICES=6 python main.py --model Qwen/Qwen2.5-0.5B --prune_method magnitude --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=7 python main.py --model Qwen/Qwen2.5-1.5B --prune_method magnitude --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-3B --prune_method magnitude --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=4 python main.py --model Qwen/Qwen2.5-7B --prune_method magnitude --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &



# CUDA_VISIBLE_DEVICES=5 python main.py --model Qwen/Qwen2.5-0.5B --prune_method wanda --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=6 python main.py --model Qwen/Qwen2.5-1.5B --prune_method wanda --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=7 python main.py --model Qwen/Qwen2.5-3B --prune_method wanda --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-7B --prune_method wanda --sparsity_ratio 0.25 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &


# CUDA_VISIBLE_DEVICES=4 python main.py --model Qwen/Qwen2.5-0.5B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=6 python main.py --model Qwen/Qwen2.5-1.5B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

# CUDA_VISIBLE_DEVICES=7 python main.py --model Qwen/Qwen2.5-3B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

CUDA_VISIBLE_DEVICES=4 python main.py --model Qwen/Qwen2.5-7B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &


CUDA_VISIBLE_DEVICES=5 python main.py --model Qwen/Qwen2.5-0.5B --prune_method wanda --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

CUDA_VISIBLE_DEVICES=6 python main.py --model Qwen/Qwen2.5-1.5B --prune_method wanda --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

CUDA_VISIBLE_DEVICES=2 python main.py --model Qwen/Qwen2.5-3B --prune_method wanda --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &

CUDA_VISIBLE_DEVICES=7 python main.py --model Qwen/Qwen2.5-7B --prune_method wanda --sparsity_ratio 0.75 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda &



# #vision

# CUDA_VISIBLE_DEVICES=1 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.5 \
    # --resume [PATH to the pretrained weights] 
