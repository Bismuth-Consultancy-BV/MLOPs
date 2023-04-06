###### EMBEDDER NODE  ######
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
import torch
import numpy

import logging
logging.disable(logging.WARNING)  

def run(input_ids, torch_device, model="openai/clip-vit-large-patch14", local_cache_only=True):
    __IDS = torch.from_numpy(numpy.array([input_ids], dtype=numpy.int64))
    #TODO: Remove duplicate tokenizer
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", local_files_only=local_cache_only)
    text_encoder.to(torch_device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(__IDS.to(torch_device))[0]

    max_length = __IDS.shape[-1]
    _tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_cache_only)
    uncond_input = _tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    _embeddings = text_embeddings.detach().cpu().numpy()

    ATTR_UNCONDITIONAL_EMBEDDING = _embeddings[0]
    ATTR_CONDITIONAL_EMBEDDING = _embeddings[1]
    ATTR_EMBEDDING_SHAPE = list(ATTR_UNCONDITIONAL_EMBEDDING.shape)
    ATTR_UNCONDITIONAL_EMBEDDING = ATTR_UNCONDITIONAL_EMBEDDING.flatten()
    ATTR_CONDITIONAL_EMBEDDING = ATTR_CONDITIONAL_EMBEDDING.flatten()

    embedding = {}
    embedding["unconditional_embedding"] = ATTR_UNCONDITIONAL_EMBEDDING
    embedding["conditional_embedding"] = ATTR_CONDITIONAL_EMBEDDING
    embedding["tensor_shape"] = ATTR_EMBEDDING_SHAPE
    return embedding 
###### EMBEDDER NODE  ######