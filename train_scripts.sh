python main.py --model deit_small_patch16_224 --data_path /home/as26840@ens.ad.etsmtl.ca/data/imagenet --resume /home/as26840@ens.ad.etsmtl.ca/weights/deit_small_patch16_224-cd65a155.pth --prune_metric wanda --prune_granularity row --sparsity 0.5

python vitslim.py --model deit_small_patch16_224 --data_path /home/as26840@ens.ad.etsmtl.ca/data/imagenet --resume /home/as26840@ens.ad.etsmtl.ca/weights/deit_small_patch16_224-cd65a155.pth --prune_metric wanda --prune_granularity row --sparsity 0.5



python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save /ephemeral/amin/aamer/logs/wanda