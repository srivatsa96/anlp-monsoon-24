from transformers import AutoTokenizer
import torch
import os
from models.transformer import EncoderDecoderTransformer
experiment_id = '24b37128-6b1f-41e2-a41a-3f8d04bd029'
expoch = 9
device = 'cuda'
MODEL_CKPT_DIRECTORY = 'trained_models'
MODEL_NAME = 'EncoderDecoderTransformer-experiment-89a6effc-3b93-490d-9333-76fc902f3ea5_epoch_14.pth'
TOKENISER_NAME ='EncoderDecoderTransformer-experiment-89a6effc-3b93-490d-9333-76fc902f3ea5_tokeniser'
start_token = '<|I_am_start|>'
tokeniser = AutoTokenizer.from_pretrained(os.path.join(MODEL_CKPT_DIRECTORY,TOKENISER_NAME))
trained_model = torch.load(os.path.join(MODEL_CKPT_DIRECTORY,MODEL_NAME)).to(device)

def generate_translation_prediction(eng_sent,tokeniser,model):
    temp=0.0000001
    #EN_SENT = 'When I was in my 20s, I saw my very first psychotherapy client.'
    en_sent_enc  = tokeniser([eng_sent], return_tensors='pt', padding=True, truncation=True,max_length=100)['input_ids'].to(device)
    FR_SENT = start_token
    FR_SENT_ENC = tokeniser([FR_SENT], return_tensors='pt', padding=True, truncation=True,max_length=50)['input_ids'].to(device)
    generated = tokeniser.decode(model.generate(en_sent_enc, FR_SENT_ENC, 100)[0].tolist(), skip_special_tokens=False)




