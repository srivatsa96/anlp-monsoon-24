import torch
from enum import Enum
import torch.nn as nn

class FinetuningMethod(Enum):
    PROMPT_TUNING = 0
    LoRA = 1
    HEAD_TUNING = 2

class FTConfig:

    def __init__(self,method):
        self.method = method

class PromptTuningConfig(FTConfig):

    def __init__(self,
                 embedding_dimension=768,
                 num_of_soft_prompt_tokens = 1,
                 initial_prompt=None,
                 tokeniser=None
                 ):
        self.method = FinetuningMethod.PROMPT_TUNING
        self.num_of_soft_prompt_tokens = num_of_soft_prompt_tokens
        self.embedding_dimension = embedding_dimension
        self.initial_prompt = initial_prompt
        self.tokeniser = tokeniser

class SummarisationModelWithSoftPrompt(nn.Module):
    def __init__(self,base_model,config):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = self.base_model.config.hidden_size

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        if config.initial_prompt is not None: 
        ## Non Random Initialisation
            if config.tokeniser is None: 
                raise ValueError("Tokeniser needed to initialise embedding from token prompts")
            ## Generate Embedding from Initial Prompt and save it as parameter
            input_ids = config.tokeniser(config.initial_prompt,return_tensors="pt")['input_ids'][0].to(base_model.device.type)
            initial_embedding = base_model.transformer.wte(input_ids)
            self.prompt_embeddings = torch.nn.Parameter(initial_embedding)
            self.prompt_length = self.prompt_embeddings.shape[0]
        else: 
        ## Random Initialiation
            if config.num_of_soft_prompt_tokens is None or config.num_of_soft_prompt_tokens < 1:
                raise ValueError("Radom Initialisation Requires Count of Token")
            self.prompt_embeddings = nn.Embedding(torch.randn(config.num_of_soft_prompt_tokens, self.hidden_size))
            self.prompt_length = config.num_of_soft_prompt_tokens
    
    def forward(self,input_ids,attention_mask,labels=None):
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        prompt_embeddings_expanded = self.prompt_embeddings.unsqueeze(0).expand(input_ids.size(0), -1, -1) # (B X Seq X Embedding Dimension)
        ## Add trainable prompt
        input_with_prompt_embeddings = torch.cat([prompt_embeddings_expanded, input_embeddings], dim=1)
        print(input_with_prompt_embeddings.shape)
        ## Pad label to account for extra prompt
        label_pad = torch.full((input_ids.size(0), self.prompt_length),-100).to(self.base_model.device.type)
        label_with_extended_pad = torch.cat([label_pad, labels],dim=1)
        print(label_with_extended_pad.shape)

        ## Concat 1s for Soft Prompt
        prompt_attention_mask = torch.ones(input_ids.size(0), self.prompt_length).to(self.base_model.device.type)
        extended_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        print(extended_attention_mask.shape)

        # Forward through the model with adjusted embeddings and mask
        return self.base_model(inputs_embeds=input_with_prompt_embeddings, attention_mask=extended_attention_mask, labels=label_with_extended_pad)





