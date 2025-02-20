from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

model_name = "Qwen/Qwen2.5-3B"
# model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')


nsamples = 128 
seed = 0 
seqlen = 512

# Generate samples from training set
random.seed(0)
trainloader = []

for _ in range(nsamples):
    while True:
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
        if trainenc.input_ids.shape[1] > seqlen:
            break
    i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = trainenc.input_ids[:, i:j]
    tar = inp.clone()
    tar[:, :-1] = -100
    trainloader.append((inp, tar))
# Prepare validation dataset
valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
valenc = valenc.input_ids[:, :(256 * seqlen)]
valenc = TokenizerWrapper(valenc)
