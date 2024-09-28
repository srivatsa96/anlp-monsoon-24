from transformers import AutoTokenizer
import torch
import os

MODEL_CKPT_DIRECTORY = 'trained_models'
MODEL_NAME = ''
TOKENISER_NAME = ''
start_token = start_token='<|I_am_start|>'
tokeniser = AutoTokenizer.from_pretrained(os.path.join(MODEL_CKPT_DIRECTORY,TOKENISER_NAME))
trained_model = torch.load(os.path.join(MODEL_CKPT_DIRECTORY,TOKENISER_NAME)).to('cuda')

EN_SENT = ''
EN_SENT_ENC  = tokeniser(EN_SENT, return_tensors='pt', padding=True, truncation=True,max_length=50)['input_ids']
FR_SENT = start_token
FR_SENT_ENC = tokeniser(EN_SENT, return_tensors='pt', padding=True, truncation=True,max_length=50)['input_ids']
generated = tokeniser.decode(trained_model.generate(EN_SENT_ENC, FR_SENT_ENC, 50), skip_special_tokens=False)
print(generated)




