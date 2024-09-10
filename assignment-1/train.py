import os 
import re
import sys
import math
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from accelerate import Accelerator
from accelerate import notebook_launcher
import wandb
import uuid

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) 
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import TrainConfig, Config
from models.lstm import LSTMLanguageModel
from models.n_gram import NGramLanguageModel
from models.transformer import TSGLangauageModel

## Constants
DATASET_PATH = '../dataset/processed/Auguste_Maquet.txt'
MODEL_CKPT_DIRECTORY = 'trained_model'
NUM_GPU = 1
READ_ORIGINAL = False


#wandb.login()


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

def generate_labeled_data(dataset_cleaned,train_size,test_size,context_size,left_pad=False):
    dataset_cleaned = ' '.join(list(map(lambda x: ' '.join(['<SOS>'] + x + ['<EOS>']),dataset_cleaned))) 

## Section 2: Read the Training Data ##

# Sliding Window Implementation
# Data Generator: Creates X,Y pairs for training where X is the  input token idices for a given context window and Y is the target indices
# Y is shifted version of X by one token when ran over entire corpus.
def get_set(data,block_size):
    x = torch.stack([data[i:i+block_size] for i in range(0,len(data)-block_size-2)])
    y = torch.stack([data[i+1:i+block_size+1] for i in range(0,len(data)-block_size-2)])
    return x,y


def load_data_and_get_pytorch_dataloaders(dataset_path, batch_size, block_size, num_gpu, encoder, clean_dataset = False):
   # print('inside loader')
    with open(dataset_path, 'r') as f:
        dataset = f.readlines()
   # print('loaded data')
    dataset_cl = None
    
    if clean_dataset:
        dataset_cl = clean_data(dataset)
    else:
        dataset_cl = ' '.join(dataset)
   # print('Read dataset')
    # Train Test Split 
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


## Section 3: Training ##

## Training Routines

'''
The following function trains a given language model with HF's Accelerate Framework using DDP.
'''
def train_loop():

    if not os.path.exists(MODEL_CKPT_DIRECTORY):
        os.makedirs(MODEL_CKPT_DIRECTORY)

    train_now = True
    ## Intialise Accelerator with W&B Logging
    accelerator = Accelerator(log_with="wandb")
    num_gpus = accelerator.num_processes  # Determine the number of GPUs/processes
    accelerator.print(f'Launching training on {num_gpus} gpu')
    resume_from_checkpoint = False
    chk_path = 'model_0.pth'

    ## Initialise Tokeniser
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l)
   # accelerator.print('Initialised tokeniser')
    ## Initialise Train and Test Config
    config = Config(tokenizer=tokenizer)
    train_config = TrainConfig(accelerator)
   # accelerator.print('Initialised config')
    try:    
        ## Loss Estimator
        @torch.no_grad()
        def estimate_loss(model, dataloader, num_gpus):
            model.eval()
            total_loss = 0
            total_iteration = 0
            for X, Y in dataloader:
                logits, loss = model(X, Y)
                loss = accelerator.gather(loss)
                total_loss += loss.sum()
                total_iteration += num_gpus  
            model.train()
            return total_loss / total_iteration
      #  accelerator.print('calling loader')       
        train_dataloader, eval_dataloader, test_dataloader = load_data_and_get_pytorch_dataloaders(DATASET_PATH,
                                                                                  config.batch_size,
                                                                                  config.block_size,
                                                                                  num_gpus,
                                                                                  encode,
                                                                                  not READ_ORIGINAL
                                                                                  )
        torch.manual_seed(42)
        print(len(train_dataloader),len(eval_dataloader))
        # Initialize or load model
        if not resume_from_checkpoint:
            model = NGramLanguageModel(config, train_config.device)
            model_name = type(model).__name__
        else:
            accelerator.load_state_dict(torch.load(chk_path))  # Use accelerator's method to load in a distributed setup
        accelerator.print('I came first')
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
     #   accelerator.print('I am here') 
        if train_now:
            experiment_name = '{0}-experiment-{1}'.format(model_name,str(uuid.uuid4()))
            accelerator.init_trackers(
                init_kwargs={"wandb": {"name": experiment_name}},
                project_name="anlp-assignment-1",
                config={ **config.__dict__, **train_config.__dict__}
            )
    #        accelerator.print('Launching training')
            for epoch in range(train_config.epochs):
                accelerator.print(f"Epoch {epoch+1}/{train_config.epochs}")
                
                # Training loop
                model.train()
                total_train_loss = 0
                for xb, yb in train_dataloader:
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    optimizer.step()
                
                # Evaluate
                train_loss = estimate_loss(model, train_dataloader,num_gpus)
                eval_loss = estimate_loss(model, eval_dataloader,num_gpus)
                
                accelerator.print(f"Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {eval_loss:.4f}")
                accelerator.log({"train_loss": train_loss, "valid_loss": eval_loss}, step=epoch+1)          
                accelerator.wait_for_everyone()
                
                # Save model checkpoint
                model_xp = accelerator.unwrap_model(model)
                accelerator.save(model_xp, os.path.join(MODEL_CKPT_DIRECTORY,f"{experiment_name}_epoch_{epoch+1}.pth"))
            test_loss = estimate_loss(model,test_dataloader,num_gpus)
            accelerator.print(f"Final Test Set loss {test_loss:.4f}")
            accelerator.end_training()
    except Exception as e:
        accelerator.print(e)
        pass
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    print('Starting training')
    wandb.login()
    train_loop()
