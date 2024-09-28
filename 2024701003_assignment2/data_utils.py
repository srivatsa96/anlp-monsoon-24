import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
        def __init__(self,X,Y):
            self.X = X
            self.Y = Y
        def __len__(self):
            return self.X['input_ids'].shape[0]
        def __getitem__(self,idx):
            return  (self.X['input_ids'][idx],
                     self.X['attention_mask'][idx],
                     self.Y['input_ids'][idx][:-1],
                     self.Y['attention_mask'][idx][:-1],
                     self.Y['input_ids'][idx][1:],
                     self.Y['attention_mask'][idx][1:])

def format_lines(Y,start_token='<|I_am_start|>',end_token='<|I_am_end|>'):
    return list(map(lambda y: start_token + y[:-1] + end_token, Y))

def generate_data_loader(X_train,Y_train,X_dev,Y_dev,X_test,Y_test,
                         batch_size = 1,
                         tokenizer=None,
                         start_token='<|I_am_start|>',
                         end_token='<|I_am_end|>',
                         max_legth=100,
                         num_gpu=0):
    def tokenise_data(X,Y,tokenizer=None,start_token='<|I_am_start|>',end_token='<|I_am_end|>',max_legth=50):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Define your special tokens
            special_tokens_dict = {
                'additional_special_tokens': [start_token,end_token]  # Start and end tokens
            }
            # Add special tokens to the tokenizer
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

            print(f"Added {num_added_toks} special tokens.")
    
        X_tokens = tokenizer(X, return_tensors='pt', padding=True, truncation=True,max_length=100)
        Y_tokens = tokenizer(format_lines(Y), return_tensors='pt', padding=True, truncation=True,max_length=100)
        return X_tokens, Y_tokens, tokenizer
    
    X_train_tokens, Y_train_tokens, tokenizer = tokenise_data(X_train,Y_train)
    X_dev_tokens, Y_dev_tokens, _ = tokenise_data(X_dev,Y_dev,tokenizer=tokenizer)
    X_test_tokens, Y_test_tokens, _ = tokenise_data(X_test,Y_test,tokenizer=tokenizer)
            
    train_dataset = TranslationDataset(X_train_tokens,Y_train_tokens)
    eval_dataset = TranslationDataset(X_dev_tokens,Y_dev_tokens)
    test_data = TranslationDataset(X_test_tokens,Y_test_tokens)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers = num_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size*2, num_workers = num_gpu)
    test_dataloader = DataLoader(test_data,shuffle=False,batch_size=batch_size*2, num_workers = num_gpu)
    return train_dataloader, eval_dataloader, test_dataloader, tokenizer
