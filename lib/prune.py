import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

# llama2
llama_seqlen = 2048 # code takes too long to run otherwise


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.config.output_attentions = True
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # shape: (128, 4096, 4096)
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    cache = {'i': 0, 'attention_mask': None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 

    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_embeddings 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_random(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    for name, param in model.named_parameters():
        print(name, param.size())
    
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # numel: total elements in tensor
                # select the threshold value at the sparsity_ratio index
                W_mask = torch.cuda.FloatTensor(W_metric.shape[0], W_metric.shape[1]).uniform_() < sparsity_ratio

            W[W_mask] = 0


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    for name, param in model.named_parameters():
        print(name, param.size())
    
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # numel: total elements in tensor
                # select the threshold value at the sparsity_ratio index
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # Save the current caching configuration and then disable caching for calibration.
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    # Load a calibration dataset ("c4") with a specified number of samples, sequence length, etc.
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    
    # Prepare calibration inputs, outputs, attention masks, and position IDs.
    # The shapes are commented for reference.
    # inps: (num_samples, sequence_length, hidden_dim)
    # outs: (num_samples, sequence_length, hidden_dim)
    # attention_mask: (1, 1, sequence_length, sequence_length)
    # position_ids: (1, sequence_length)
    with torch.no_grad():
        # inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        inps, outs, attention_mask, position_embeddings = prepare_calibration_input(model, dataloader, device)

    # Get all transformer layers from the model.
    layers = model.model.layers
    for i in range(len(layers)):
        # Process each transformer layer one by one.
        layer = layers[i]
        
        # Extract the target submodules (e.g., specific linear layers) within the current transformer layer.
        subset = find_layers(layer)

        # In multi-GPU setups (e.g., for very large models like llama-30B/65B), move calibration data
        # to the device that the current layer is mapped to.
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
            cos, sin = position_embeddings
            cos, sin = cos.to(dev), sin.to(dev)
            position_embeddings = (cos, sin)

        # Create a dictionary to hold WrappedGPT instances for each target submodule.
        # Each WrappedGPT instance collects activation statistics during forward passes.
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        # Define a hook function that will be called during the forward pass of a submodule.
        def add_batch(name):
            # The inner function "tmp" will be attached as a forward hook.
            def tmp(_, inp, out):
                # Use the first element of the input (assumed to be the relevant tensor) and the output,
                # and update the running statistics in the corresponding WrappedGPT instance.
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register the forward hooks for each target submodule.
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Run forward passes on the calibration data to collect activation statistics.
        for j in range(args.nsamples):
            with torch.no_grad():
                # Unsqueeze the input to add a batch dimension and process it through the current layer.
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                #                  position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                 position_embeddings=position_embeddings)[0]
        
        # Remove the forward hooks after the statistics have been collected.
        for h in handles:
            h.remove()
                
        # For each submodule in the current layer, compute a pruning metric and apply pruning.
        for name in subset:
            print(f"pruning layer {i} name {name}")
            # Compute a metric for pruning based on the absolute value of the weights
            # and the activation statistics (scaled by the square root of scaler_row).
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Initialize a mask with the same shape as W_metric where all entries are False.
            # This mask will mark weights for pruning (True means prune that weight).
            W_mask = (torch.zeros_like(W_metric) == 1)
                        
            if prune_n != 0:
                # Perform structured n:m pruning if prune_n is provided.
                # For each group (block) of prune_m weights, prune the prune_n weights with the smallest metric.
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        # Get a block of prune_m columns starting at the current index.
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        # Identify the indices of the prune_n smallest elements in the block.
                        indices_to_prune = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        # Update the mask to mark these indices as True (pruned).
                        W_mask.scatter_(1, ii + indices_to_prune, True)
            else:
                # For unstructured pruning, first sort the metric values.
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # Wanda variant: adjust the pruning threshold (alpha) using binary search
                    # to achieve a target sparsity ratio.
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    # Get the initial mask and current sparsity using the given alpha.
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    # Adjust alpha until the achieved sparsity is within a small tolerance of the target.
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # Simple unstructured pruning: prune a fixed fraction of weights by selecting
                    # the indices corresponding to the smallest metric values.
                    # for a layer of seq_length x n_embd, we prune n_embd * sparsity_ratio weights for each element in the sequence
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            # Apply the mask: set the selected weights to zero (prune them).
            subset[name].weight.data[W_mask] = 0  
                
        # After pruning the current layer, run the forward pass again on the calibration data.
        # This updates the activations so they can serve as inputs for the next layer.
        for j in range(args.nsamples):
            with torch.no_grad():
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                #                  position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                 position_embeddings=position_embeddings)[0]                
        # Swap inps and outs: the output of this layer becomes the input for the next layer.
        inps, outs = outs, inps

    # Restore the original caching configuration.
    model.config.use_cache = use_cache 
    # Clear the CUDA cache to free up memory.
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    cache = {'i': 0, 'attention_mask': None, "position_embeddings": None}


    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()