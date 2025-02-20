# CUDA_VISIBLE_DEVICES=1 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.5 &

# CUDA_VISIBLE_DEVICES=2 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.25 &

# CUDA_VISIBLE_DEVICES=3 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0 &

# CUDA_VISIBLE_DEVICES=4 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0.25 &

# CUDA_VISIBLE_DEVICES=5 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0 &

# CUDA_VISIBLE_DEVICES=6 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0.5 &

# CUDA_VISIBLE_DEVICES=2 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.75 &

# CUDA_VISIBLE_DEVICES=3 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0.75 &



# CUDA_VISIBLE_DEVICES=1 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.5 &

# CUDA_VISIBLE_DEVICES=4 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0.25 &

# CUDA_VISIBLE_DEVICES=5 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric wanda --prune_granularity row --sparsity 0 &

# CUDA_VISIBLE_DEVICES=6 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0.25 &

# CUDA_VISIBLE_DEVICES=2 python main.py --model deit_base_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0 &

# CUDA_VISIBLE_DEVICES=3 python main.py --model deit_small_patch16_224 --data_path /home/aamer/data/ImageNet --prune_metric magnitude --prune_granularity row --sparsity 0.5 &


