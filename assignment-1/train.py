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

from data_utils import load_data_and_get_pytorch_dataloaders

## Constants
DATASET_PATH = '../dataset/processed/Auguste_Maquet.txt'
MODEL_CKPT_DIRECTORY = 'trained_model'
NUM_GPU = 1
READ_ORIGINAL = False
SENTENCE_LEVEL_MODELLING = True



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
                                                                                  not READ_ORIGINAL,
                                                                                  not SENTENCE_LEVEL_MODELLING
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
