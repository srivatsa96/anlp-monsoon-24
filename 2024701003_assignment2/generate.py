import os
import torch
import csv
import fire
from tqdm import tqdm
from transformers import AutoTokenizer
from models.transformer import EncoderDecoderTransformer
from torchtext.data.metrics import bleu_score

def generate_translation_prediction(eng_sents, tokeniser, model, device):
    # Tokenize the input sentences and create attention masks
    en_sent_enc = tokeniser(eng_sents, return_tensors='pt', padding=True, truncation=True, max_length=100)
    input_ids = en_sent_enc['input_ids'].to(device)
    attention_mask = en_sent_enc['attention_mask'].to(device)
    
    FR_SENT = '<|I_am_start|>'
    FR_SENT_ENC = tokeniser([FR_SENT], return_tensors='pt', padding=True, truncation=True, max_length=50)['input_ids'].to(device)

    # Generate translations in a batch
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, decoder_input_ids=FR_SENT_ENC, max_length=100)
    
    # Decode the generated sentences
    generated_sentences = tokeniser.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_sentences

def evaluate(X, Y, tokeniser, model, device, csv_file, batch_size):
    with open(csv_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Sentence No', 'English Sentence', 'Reference French', 'Predicted French', 'BLEU Score'])
        
        references = []
        hypotheses = []
        
        # Wrap the sentence processing loop with tqdm for progress display
        for i in tqdm(range(0, len(X), batch_size), desc="Evaluating Sentences"):  # Use specified batch size
            batch_en = X[i:i+batch_size]
            batch_fr = Y[i:i+batch_size]

            fr_preds = generate_translation_prediction([en.strip() for en in batch_en], tokeniser, model, device)
            
            for en, fr, fr_pred in zip(batch_en, batch_fr, fr_preds):
                # Tokenize and append to lists for computing BLEU score
                reference = [tokeniser.tokenize(fr.strip())]
                hypothesis = tokeniser.tokenize(fr_pred)
                references.append(reference)
                hypotheses.append(hypothesis)
                
                # Compute per-sentence BLEU score
                sentence_bleu = bleu_score([hypothesis], [reference])
                
                # Write per-sentence results to CSV
                writer.writerow([i + 1, en.strip(), fr.strip(), fr_pred, f"{sentence_bleu:.4f}"])
        
        # Compute overall BLEU score
        overall_bleu = bleu_score(hypotheses, references)
        writer.writerow(['Overall BLEU', '', '', '', f"{overall_bleu:.4f}"])
        print(f"Overall BLEU score: {overall_bleu:.4f}")

def main(experiment_id=None, epoch=None, set_type=None, eng_sent=None, batch_size=32):
    """
    Arguments:
    - experiment_id: str, The experiment ID of the model.
    - epoch: int, The epoch of the model to load.
    - set_type: str, The dataset to use: one of 'train', 'test', 'eval'. Optional if using interactive mode.
    - eng_sent: str, English sentence to translate in interactive mode. If provided, the script will translate this sentence and output the result.
    - batch_size: int, The size of the batches for translation. Default is 32.

    Example usage:
    Interactive Mode:
    python generate.py --experiment_id=24b37128 --epoch=9 --eng_sent="How are you?" --batch_size=16
    
    Evaluation Mode:
    python generate.py --experiment_id=24b37128 --epoch=9 --set_type=test --batch_size=16
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    MODEL_CKPT_DIRECTORY = 'trained_models'
    MODEL_NAME = f'EncoderDecoderTransformer-experiment-{experiment_id}_epoch_{epoch}.pth'
    TOKENISER_NAME = f'EncoderDecoderTransformer-experiment-{experiment_id}_tokeniser'
    
    tokeniser = AutoTokenizer.from_pretrained(os.path.join(MODEL_CKPT_DIRECTORY, TOKENISER_NAME))
    trained_model = torch.load(os.path.join(MODEL_CKPT_DIRECTORY, MODEL_NAME)).to(device)
    
    # Interactive mode: Translate a single sentence
    if eng_sent:
        fr_pred = generate_translation_prediction([eng_sent.strip()], tokeniser, trained_model, device)[0]
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
        
        # Evaluate and write to CSV
        evaluate(X_TEST, Y_TEST, tokeniser, trained_model, device, csv_file, batch_size)
    else:
        print("Error: Either provide a sentence for interactive mode (using --eng_sent) or specify set_type for evaluation mode.")
    
if __name__ == "__main__":
    fire.Fire(main)
