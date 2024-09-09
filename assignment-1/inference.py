import os 
import torch
from transformers import GPT2Tokenizer

MODEL_PATH = 'TSGLangauageModel-experiment-b0b32ad1-ba9d-4ea1-b520-6f645d561bbb_epoch_9.pth'
NUM_TOKENS = 100
PROMPT = 'The world is but'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)


# generate from the model
trained_model = torch.load(MODEL_PATH)
trained_model.eval()
prompt_idx = torch.unsqueeze(torch.Tensor(encode(PROMPT)),0).long().to('cuda')
generated_text = decode(trained_model.generate(prompt_idx, max_new_tokens=NUM_TOKENS)[0].tolist())
print(generated_text)