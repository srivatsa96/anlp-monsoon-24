from tqdm.tqdm import tqdm
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from data_utils import clean_data_return_sentence, get_set

dataset_path = '../dataset/processed/Auguste_Maquet.txt'
clean_dataset = True
sentence_level_modelling = True

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

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
    data = list(map(lambda x: torch.tensor(encode(' '.join(x)), dtype=torch.long), dataset_cl))
    data = list(filter(lambda x: x.shape[0]<52,data))
else:
    data = torch.tensor(encode(dataset_cl), dtype=torch.long)

n = int(0.7*len(data)) 
e = int(0.9*len(data))
train_data = data[:n]  # 70% Data for Training
eval_data = data[n:e]  # 20% Data for Evaluation
test_data = data[e:]   # 10% Data for Testing

## X,Y Pairs
train_set = get_set(train_data,50,True)
eval_set = get_set(eval_data,50,True)
test_set = get_set(test_data,50,True)


trained_model = torch.load('model_epoch_9.pth').to('cuda')
@torch.no_grad()
def estimate_test_loss(model, dataset,filename,decoder):
    with torch.device('cuda'):
        model.eval()
        total_loss = 0
        total_iteration = 0
        with open(filename,'a') as f:
            for idx in tqdm(range(0,len(dataset[0]))):
                X = torch.unsqueeze(dataset[0][idx],dim=0).to('cuda')
                Y = torch.unsqueeze(dataset[1][idx],dim=0).to('cuda')
                _, loss = model(X, Y)
                total_loss += loss
                f.write(f'{idx}/{len(dataset[0])}\t{decoder(X.cpu().detach().numpy()[0])}\t{loss.item()}\n')
                total_iteration += 1  
            model.train()
            avg_loss = total_loss / total_iteration
            print(avg_loss.item())
            f.write(f'Average Loss {0}'.format(avg_loss.item()))
        return avg_loss

estimate_test_loss(trained_model,train_set,'2024701003-Transformers-Train-Perplexity.txt')
estimate_test_loss(trained_model,test_set,'2024701003-Transformers-Test-Perplexity.txt')
estimate_test_loss(trained_model,eval_set,'2024701003-Transformers-Val-Perplexity.txt')