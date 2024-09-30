import os
import torch
import csv
import fire
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from models.transformer import EncoderDecoderTransformer
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.multiprocessing as mp

# Dataset class to handle batched input
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(self.src_texts[idx], return_tensors='pt', padding=True, truncation=True, max_length=100)
        tgt_enc = self.tokenizer(self.tgt_texts[idx], return_tensors='pt', padding=True, truncation=True, max_length=50)
        return src_enc['input_ids'].squeeze(), src_enc['attention_mask'].squeeze(), tgt_enc['input_ids'].squeeze()

# Generate translations in batch
def generate_translation_batch(src_batch, src_attention_mask, tokenizer, model, device, search_mode, beam_width=5):
    batch_size = src_batch.shape[0]
    FR_SENT = '<|I_am_start|>'
    fr_start = tokenizer([FR_SENT] * batch_size, return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids'].to(device)
    
    if search_mode == 'beam':
        return model.generate(src_batch, fr_start, max_new_tokens=100, beam_width=beam_width, mode='beam', enc_attention_mask=src_attention_mask)
    else:
        return model.generate(src_batch, fr_start, max_new_tokens=100, mode='standard', enc_attention_mask=src_attention_mask)

# Generate translation for a single sentence (interactive mode)
def generate_translation_prediction(eng_sent, tokenizer, model, device, search_mode='standard', beam_width=5):
    src_enc = tokenizer([eng_sent], return_tensors='pt', padding=True, truncation=True, max_length=100)
    en_sent_enc = src_enc['input_ids'].to(device)
    enc_attention_mask = src_enc['attention_mask'].to(device)

    FR_SENT = '<|I_am_start|>'
    fr_start = tokenizer([FR_SENT], return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids'].to(device)

    if search_mode == 'beam':
        generated = model.generate(en_sent_enc, fr_start, max_new_tokens=100, beam_width=beam_width, mode='beam', enc_attention_mask=enc_attention_mask)
    else:
        generated = model.generate(en_sent_enc, fr_start, max_new_tokens=100, mode='standard', enc_attention_mask=enc_attention_mask)

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

# Evaluation loop for BLEU score calculation
def evaluate(rank, world_size, X, Y, tokenizer, model, device, csv_file, search_mode, batch_size=32):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = DDP(model, device_ids=[rank])
    
    dataset = TranslationDataset(X, Y, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with open(csv_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Sentence No', 'English Sentence', 'Reference French', 'Predicted French', 'BLEU Score'])

        references = []
        hypotheses = []
        
        for i, (src_batch, src_attention_mask, tgt_batch) in enumerate(tqdm(data_loader, desc="Evaluating Sentences")):
            src_batch = src_batch.to(device)
            src_attention_mask = src_attention_mask.to(device)
            tgt_batch = tgt_batch.to(device)

            fr_pred_batch = generate_translation_batch(src_batch, src_attention_mask, tokenizer, model, device, search_mode)
            fr_pred_texts = [tokenizer.decode(pred.tolist(), skip_special_tokens=True) for pred in fr_pred_batch]
            tgt_texts = [tokenizer.decode(tgt.tolist(), skip_special_tokens=True) for tgt in tgt_batch]
            
            for idx, (hypothesis, reference) in enumerate(zip(fr_pred_texts, tgt_texts)):
                reference_tokenized = [tokenizer.tokenize(reference.strip())]
                hypothesis_tokenized = tokenizer.tokenize(hypothesis)
                references.append(reference_tokenized)
                hypotheses.append(hypothesis_tokenized)
                
                # Compute per-sentence BLEU score
                sentence_bleu = bleu_score([hypothesis_tokenized], [reference_tokenized])
                writer.writerow([i * batch_size + idx + 1, X[i * batch_size + idx].strip(), Y[i * batch_size + idx].strip(), hypothesis, f"{sentence_bleu:.4f}"])

        # Compute overall BLEU score
        overall_bleu = bleu_score(hypotheses, references)
        writer.writerow(['Overall BLEU', '', '', '', f"{overall_bleu:.4f}"])
        print(f"Overall BLEU score: {overall_bleu:.4f}")
    
    dist.destroy_process_group()

def setup_ddp(rank, world_size, X, Y, tokenizer, model, csv_file, search_mode, batch_size):
    # Setting up the environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Each rank should use a different GPU
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    
    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    # Perform evaluation
    evaluate(rank, world_size, X, Y, tokenizer, model, device, csv_file, search_mode, batch_size)
    
    # Cleanup the process group
    dist.destroy_process_group()

def main(experiment_id=None, epoch=None, set_type=None, eng_sent=None, n_gpu=1, search_mode='standard', batch_size=32):
    """
    Arguments:
    - experiment_id: str, The experiment ID of the model.
    - epoch: int, The epoch of the model to load.
    - set_type: str, The dataset to use: one of 'train', 'test', 'eval'. Optional if using interactive mode.
    - eng_sent: str, English sentence to translate in interactive mode.
    - n_gpu: int, Number of GPUs to use for evaluation. Defaults to 1 (single GPU).
    - search_mode: str, Type of search to use for generation: 'standard' or 'beam'. Defaults to 'standard'.
    - batch_size: int, Batch size for evaluation. Defaults to 32.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    MODEL_CKPT_DIRECTORY = 'trained_models'
    MODEL_NAME = f'EncoderDecoderTransformer-experiment-{experiment_id}_epoch_{epoch}.pth'
    TOKENISER_NAME = f'EncoderDecoderTransformer-experiment-{experiment_id}_tokeniser'
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_CKPT_DIRECTORY, TOKENISER_NAME))
    trained_model = torch.load(os.path.join(MODEL_CKPT_DIRECTORY, MODEL_NAME)).to(device)

    # Interactive mode: Translate a single sentence
    if eng_sent:
        fr_pred = generate_translation_prediction(eng_sent.strip(), tokenizer, trained_model, device, search_mode)
        print(f"English Sentence: {eng_sent.strip()}")
        print(f"Translated French Sentence: {fr_pred}")
    
    # Evaluation mode: Evaluate a set of sentences
    elif set_type:
        data_path = f'../dataset/ted-talks-corpus/{set_type}.'
        TEST_DATA_PATH_X = data_path + 'en'
        TEST_DATA_PATH_Y = data_path + 'fr'
        
        with open(TEST_DATA_PATH_X, 'r') as f:
            X_TEST = f.readlines()
        with open(TEST_DATA_PATH_Y, 'r') as f:
            Y_TEST = f.readlines()

        # Define output CSV file name
        csv_file = f'bleu_scores_{experiment_id}_epoch_{epoch}_{set_type}.csv'
        
        # If using multiple GPUs, initialize DDP
        if n_gpu > 1:
            world_size = n_gpu
            mp.spawn(setup_ddp, args=(world_size, X_TEST, Y_TEST, tokenizer, trained_model, device, csv_file, search_mode, batch_size), nprocs=world_size, join=True)
        else:
            # Single GPU Evaluation
            evaluate(0, 1, X_TEST, Y_TEST, tokenizer, trained_model, device, csv_file, search_mode, batch_size)
    else:
        print("Error: Either provide a sentence for interactive mode (using --eng_sent) or specify set_type for evaluation mode.")

if __name__ == "__main__":
    fire.Fire(main)