import torch
from tqdm import notebook
from datasets import Dataset

'''
Preprocess the Data for Causal LM.
Generate InputIDs , Labels and Attention Mask
Steps:
1. Tokensises the entire dataset for summary and highlights seprately
2. Concatenates the Summary And Highlights
3. Creates Label by setting same as input ids but masking the Prompt sequence by -100 (Label shifting by 1 is taken care by model itself)
4. Pads All the inputs to max length
5. Converts to HF Dataset
'''
def preprocess_function(examples, tokenizer, max_length=1000, set_type='train', padding_token_id=0):
    batch_size = len(examples["article"])  # Use the "article" column for input size
    inputs = [f"{x} Summary : " for x in notebook.tqdm(examples["article"])]  # Format the input with "Article" and "Summary"
    targets = examples["highlights"].to_list()  # Use the "highlights" column for targets

    model_inputs = tokenizer(inputs)  # Tokenize the inputs (articles with "Summary" prompt)
    labels = tokenizer(targets, add_special_tokens=False)  # Tokenize the summaries without special tokens

    for i in notebook.tqdm(range(batch_size)):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]  # Add EOS token at the end of the summary
        
        # Concatenate input (article) with the summary and prepare input IDs for the model
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids  # Mask the article part with -100

        # Set the attention mask for the concatenated sequence
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in notebook.tqdm(range(batch_size)):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        
        # Pad the input and labels to the maximum length with right padding
        model_inputs["input_ids"][i] = sample_input_ids + [padding_token_id] * (max_length - len(sample_input_ids) - len(label_input_ids))
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [0] * (max_length - len(model_inputs["input_ids"][i]))
        labels["input_ids"][i] = labels["input_ids"][i] + [-100] * (max_length - len(labels["input_ids"][i]))

        # Truncate or pad all sequences to the max length
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    
    model_inputs["labels"] = labels["input_ids"]  # Add the labels (tokenized summaries) to the model inputs
    return Dataset.from_dict(model_inputs)

# Define a function to convert features to tensors
def convert_to_tensors(batch):
    # Convert input features to tensors
    return {
        'input_ids': torch.tensor(batch['input_ids']),
        'attention_mask': torch.tensor(batch['attention_mask']),
        'labels': torch.tensor(batch['label'])
    }

