python main.py --model deit_small_patch16_224 --data_path /home/as26840@ens.ad.etsmtl.ca/data/imagenet --resume /home/as26840@ens.ad.etsmtl.ca/weights/deit_small_patch16_224-cd65a155.pth --prune_metric wanda --prune_granularity row --sparsity 0.5

python vitslim.py --model deit_small_patch16_224 --data_path /home/as26840@ens.ad.etsmtl.ca/data/imagenet --resume /home/as26840@ens.ad.etsmtl.ca/weights/deit_small_patch16_224-cd65a155.pth --prune_metric wanda --prune_granularity row --sparsity 0.5



CUDA_VISIBLE_DEVICES=4 python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /home/aamer/repos/logs/wanda

TinyLlama/TinyLlama-1.1B-Chat-v1.0

Qwen/Qwen2.5-1.5B
meta-llama/Llama-2-7b-hf
meta-llama/Llama-3.2-1B