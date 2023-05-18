import logging
import numpy
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
import torch
from compel import Compel

logging.disable(logging.WARNING)

def run(input_prompt_positive, input_prompt_negative, torch_device, model, local_cache_only=True):
    try:
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", local_files_only=local_cache_only)
    except OSError:
        text_encoder = CLIPTextModel.from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")
    text_encoder.to(torch_device)

    try:
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_cache_only)
    except OSError as error:
        tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")

    compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder)
    conditional_embeddings = compel_proc(input_prompt_positive)
    unconditional_embeddings = compel_proc(input_prompt_negative)
    embeddings_shape = list(unconditional_embeddings.shape)[1:]
    unconditional_embeddings = unconditional_embeddings.flatten()
    conditional_embeddings = conditional_embeddings.flatten()

    embeddings = {}
    embeddings["unconditional_embedding"] = unconditional_embeddings
    embeddings["conditional_embedding"] = conditional_embeddings
    embeddings["tensor_shape"] = embeddings_shape
    return embeddings
