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

from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) 
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import TrainConfig, Config
from models.transformer import EncoderDecoderTransformer

from data_utils import generate_data_loader


## Data Paths
TRAIN_DATA_PATH_X = '../dataset/ted-talks-corpus/train.en'
TRAIN_DATA_PATH_Y = '../dataset/ted-talks-corpus/train.fr'
DEV_DATA_PATH_X = '../dataset/ted-talks-corpus/dev.en'
DEV_DATA_PATH_Y = '../dataset/ted-talks-corpus/dev.fr'
TEST_DATA_PATH_X = '../dataset/ted-talks-corpus/test.en'
TEST_DATA_PATH_Y = '../dataset/ted-talks-corpus/test.fr'
MODEL_CKPT_DIRECTORY = 'trained_models'


## Section 3: Training ##
## Training Routines

'''
The following function trains a given language model with HF's Accelerate Framework using DDP.
'''
def train_loop():

    if not os.path.exists(MODEL_CKPT_DIRECTORY):
        os.makedirs(MODEL_CKPT_DIRECTORY)

    ## Load Dataset
    # Read Data
    with open(TRAIN_DATA_PATH_X, 'r') as f:
        X_TRAIN = f.readlines()
    with open(TRAIN_DATA_PATH_Y, 'r') as f:
        Y_TRAIN = f.readlines()
    with open(DEV_DATA_PATH_X, 'r') as f:
        X_DEV = f.readlines()
    with open(DEV_DATA_PATH_Y, 'r') as f:
        Y_DEV = f.readlines()
    with open(TEST_DATA_PATH_X, 'r') as f:
        X_TEST = f.readlines()
    with open(TEST_DATA_PATH_Y, 'r') as f:
        Y_TEST = f.readlines()

    train_now = True 
    ## Intialise Accelerator with W&B Logging
    accelerator = Accelerator(log_with="wandb")
    train_config = TrainConfig(accelerator)   
    train_dataloader, eval_dataloader, test_dataloader,tokeniser = generate_data_loader(X_TRAIN,Y_TRAIN,
                                                                                        X_DEV,Y_DEV,
                                                                                        X_TEST,Y_TEST,
                                                                                        batch_size=train_config.batch_size)
    model_config = Config(tokeniser=tokeniser)


    num_gpus = accelerator.num_processes  # Determine the number of GPUs/processes
    accelerator.print(f'Launching training on {num_gpus} gpu')

    ## If resuming set checkpoint path
    resume_from_checkpoint = False
    chk_path = 'model_0.pth'

    try:    
        ## Loss Estimator
        @torch.no_grad()
        def estimate_loss(model, dataloader, num_gpus):
            dataloader_with_bar = tqdm(
                    dataloader, disable=(not accelerator.is_local_main_process)
            )
            model.eval()
            total_loss = 0
            total_iteration = 0
            for x_enc, x_enc_mask, x_dec, x_dec_mask, y, y_mask in dataloader_with_bar:
                logits, loss = model(x_enc, x_dec, y, x_enc_mask, x_dec_mask, y_mask)
                loss = accelerator.gather(loss)
                total_loss += loss.sum()
                total_iteration += num_gpus  
            model.train()
            return total_loss / total_iteration
        
        torch.manual_seed(42)
        
        # Initialize or load model
        if not resume_from_checkpoint:
            model = EncoderDecoderTransformer(model_config, train_config.device)
            model_name = type(model).__name__
        else:
            accelerator.load_state_dict(torch.load(chk_path))  # Use accelerator's method to load in a distributed setup
        

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
        model, optimizer, train_dataloader, eval_dataloader,test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader,test_dataloader
        )

        if train_now:
            experiment_name = '{0}-experiment-{1}'.format(model_name,str(uuid.uuid4()))
            tokeniser.save_pretrained(os.path.join(MODEL_CKPT_DIRECTORY,f"{experiment_name}_tokeniser"))
            accelerator.init_trackers(
                init_kwargs={"wandb": {"name": experiment_name}},
                project_name="anlp-assignment-2",
                config={ **model_config.__dict__, **train_config.__dict__}
            )

            for epoch in range(train_config.epochs):
                accelerator.print(f"Epoch {epoch+1}/{train_config.epochs}")

                # Evaluate
                train_loss = estimate_loss(model, train_dataloader,num_gpus)
                eval_loss = estimate_loss(model, eval_dataloader,num_gpus)
                
                accelerator.print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {eval_loss:.4f}")
                accelerator.log({"train_loss": train_loss, "valid_loss": eval_loss}, step=epoch+1)          
                
                # Training loop
                model.train()
                total_train_loss = 0
                dataloader_with_bar = tqdm(
                    train_dataloader, disable=(not accelerator.is_local_main_process)
                )
                for x_enc, x_enc_mask, x_dec, x_dec_mask, y, y_mask in dataloader_with_bar:
                    logits, loss = model(x_enc, x_dec, y, x_enc_mask, x_dec_mask, y_mask)
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    optimizer.step()
                
                
                accelerator.wait_for_everyone()
                
                # Save model checkpoint
                model_xp = accelerator.unwrap_model(model)
                accelerator.save(model_xp, os.path.join(MODEL_CKPT_DIRECTORY,f"{experiment_name}_epoch_{epoch+1}.pth"))
            
            test_loss = estimate_loss(model,test_dataloader,num_gpus)
            accelerator.print(f"Final Test Set loss {test_loss:.4f}")
            accelerator.end_training()
    except Exception as e:
        raise e
        pass
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    print('Starting training')
    wandb.login()
    train_loop()
