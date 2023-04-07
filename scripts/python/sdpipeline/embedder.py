###### EMBEDDER NODE  ######
from transformers import CLIPTextModel
import torch
import numpy
import logging
logging.disable(logging.WARNING)  

def run(input_ids_positive, input_ids_negative, torch_device, model="openai/clip-vit-large-patch14", local_cache_only=True):
    try:
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", local_files_only=local_cache_only)
    except OSError as error:
        text_encoder = CLIPTextModel.from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    text_encoder.to(torch_device)
    
    with torch.no_grad():
        __IDS = torch.from_numpy(numpy.array([input_ids_positive], dtype=numpy.int64))
        positive_text_embeddings = text_encoder(__IDS.to(torch_device))[0]

        __IDS2 = torch.from_numpy(numpy.array([input_ids_negative], dtype=numpy.int64))
        negative_text_embeddings = text_encoder(__IDS2.to(torch_device))[0]

    all_text_embeddings = torch.cat([negative_text_embeddings, positive_text_embeddings]).detach().cpu().numpy()

    ATTR_UNCONDITIONAL_EMBEDDING = all_text_embeddings[0]
    ATTR_CONDITIONAL_EMBEDDING = all_text_embeddings[1]
    ATTR_EMBEDDING_SHAPE = list(ATTR_UNCONDITIONAL_EMBEDDING.shape)
    ATTR_UNCONDITIONAL_EMBEDDING = ATTR_UNCONDITIONAL_EMBEDDING.flatten()
    ATTR_CONDITIONAL_EMBEDDING = ATTR_CONDITIONAL_EMBEDDING.flatten()

    embedding = {}
    embedding["unconditional_embedding"] = ATTR_UNCONDITIONAL_EMBEDDING
    embedding["conditional_embedding"] = ATTR_CONDITIONAL_EMBEDDING
    embedding["tensor_shape"] = ATTR_EMBEDDING_SHAPE
    return embedding
