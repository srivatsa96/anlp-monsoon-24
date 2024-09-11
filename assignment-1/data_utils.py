import os 
import re
import sys
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import uuid

## Section 1:  Data Cleaning Functions ##

def remove_non_alphanumeric(text):
    # Use regular expression to keep alphanumeric characters, spaces, tabs, and periods, and remove newlines
    cleaned_text = re.sub(r'[^\w\s\t.]', '', text)
    # Remove newline characters
    cleaned_text = cleaned_text.replace('\n', '')
    if cleaned_text == '':
        cleaned_text = ' '
    return cleaned_text

def find_chapter_indices(list_of_lists):
    indices = []
    for i, sublist in enumerate(list_of_lists):
        # Check if 'chapter' is in any of the elements of the sublist
        if any("chapter" in str(element).lower() for element in sublist):
            indices.append(i)
    return indices

def delete_elements_at_indices(list_of_lists, indices):
    return [sublist for i, sublist in enumerate(list_of_lists) if i not in indices]

def delete_elements_of_len_less_than_k(list_of_lists, k):
    return [sublist for i, sublist in enumerate(list_of_lists) if len(sublist)>k]

def clean_data(dataset):
    dataset_cleaned = list(map(lambda x: remove_non_alphanumeric(x),dataset))
    dataset_cleaned = (' ').join(dataset_cleaned)
    dataset_cleaned = dataset_cleaned.split('.')
    dataset_cleaned = list(map(lambda x: x.strip(),dataset_cleaned))
    dataset_cleaned = list(map(lambda x: x.split(' '),dataset_cleaned))
    dataset_cleaned = delete_elements_at_indices(dataset_cleaned,find_chapter_indices(dataset_cleaned))
    dataset_cleaned = delete_elements_of_len_less_than_k(dataset_cleaned,6)
    dataset_cleaned = '. '.join(list(map(lambda x: ' '.join(x), dataset_cleaned)))
    return dataset_cleaned

def clean_data_return_sentence(dataset):
    dataset_cleaned = list(map(lambda x: remove_non_alphanumeric(x),dataset))
    dataset_cleaned = (' ').join(dataset_cleaned)
    dataset_cleaned = dataset_cleaned.split('.')
    dataset_cleaned = list(map(lambda x: x.strip(),dataset_cleaned))
    dataset_cleaned = list(map(lambda x: x.split(' '),dataset_cleaned))
    dataset_cleaned = delete_elements_at_indices(dataset_cleaned,find_chapter_indices(dataset_cleaned))
    dataset_cleaned = delete_elements_of_len_less_than_k(dataset_cleaned,6)
    #dataset_cleaned = '. '.join(list(map(lambda x: ' '.join(x), dataset_cleaned)))
    return dataset_cleaned

def generate_labeled_data(dataset_cleaned,train_size,test_size,context_size,left_pad=False):
    dataset_cleaned = ' '.join(list(map(lambda x: ' '.join(['<SOS>'] + x + ['<EOS>']),dataset_cleaned))) 

## Section 2: Read the Training Data ##

# Sliding Window Implementation
# Data Generator: Creates X,Y pairs for training where X is the  input token idices for a given context window and Y is the target indices
# Y is shifted version of X by one token when ran over entire corpus.
def get_set(data,block_size,sentence_level=False):
    if not sentence_level:
        x = torch.stack([data[i:i+block_size] for i in range(0,len(data)-block_size-2)])
        y = torch.stack([data[i+1:i+block_size+1] for i in range(0,len(data)-block_size-2)])
        return x,y
    else:
        x = [data[i][:-1] for i in range(len(data))]
        y = [data[i][1:] for i in range(len(data))]
        return x,y


def load_data_and_get_pytorch_dataloaders(dataset_path, batch_size, block_size, num_gpu, encoder, clean_dataset = False,sentence_level_modelling=True):
    dataset_cl = None
    with open(dataset_path, 'r') as f:
        dataset = f.readlines()
    if clean_dataset:
        if sentence_level_modelling:
            dataset_cl = clean_data_return_sentence(dataset)
        else:
            dataset_cl = clean_data_return_sentence(dataset)
    else:
        dataset_cl = ' '.join(dataset)
    
    # Train Test Split 
    if  sentence_level_modelling:
        data = list(map(lambda x: torch.tensor(encoder(' '.join(x)), dtype=torch.long), dataset_cl))
        data = list(filter(lambda x: x.shape[0]<52,data))
    else:
        data = torch.tensor(encoder(dataset_cl), dtype=torch.long)
    
    n = int(0.7*len(data)) 
    e = int(0.9*len(data))
    train_data = data[:n]  # 70% Data for Training
    eval_data = data[n:e]  # 20% Data for Evaluation
    test_data = data[e:]   # 10% Data for Testing
    #print('Splitted')

    # Create Dataloaders
    train_set = get_set(train_data,block_size)
    eval_set = get_set(eval_data,block_size)
    test_set = get_set(test_data,block_size)
   # print('Created pairs')
    class TSGDatasetTrain(Dataset):
        def __len__(self):
            return len(train_set[0])
        def __getitem__(self,idx):
            return train_set[0][idx],train_set[1][idx]

    class TSGDatasetEval(Dataset):
        def __len__(self):
            return len(eval_set[0])
        def __getitem__(self,idx):
            return eval_set[0][idx],eval_set[1][idx]

    class TSGTestDataset(Dataset):
        def __len__(self):
            return len(test_set[0])
        def __getitem__(self,idx):
            return test_set[0][idx],test_set[1][idx]
        
    #print('created dataset class')
    train_dataset = TSGDatasetTrain()
    eval_dataset = TSGDatasetEval()
    test_data = TSGTestDataset()
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers = num_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size*2, num_workers = num_gpu)
    test_dataloader = DataLoader(test_data,shuffle=False,batch_size=batch_size*2, num_workers = num_gpu)
    return train_dataloader, eval_dataloader, test_dataloader