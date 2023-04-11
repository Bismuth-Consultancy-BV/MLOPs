import logging
import numpy
from transformers import CLIPTextModel
import torch

logging.disable(logging.WARNING)

def run(input_ids_positive, input_ids_negative, torch_device, model, local_cache_only=True):
    try:
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", local_files_only=local_cache_only)
    except OSError:
        text_encoder = CLIPTextModel.from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")
    text_encoder.to(torch_device)

    with torch.no_grad():
        input_ids_positive = torch.from_numpy(numpy.array([input_ids_positive], dtype=numpy.int64))
        positive_text_embeddings = text_encoder(input_ids_positive.to(torch_device))[0]

        input_ids_negative = torch.from_numpy(numpy.array([input_ids_negative], dtype=numpy.int64))
        negative_text_embeddings = text_encoder(input_ids_negative.to(torch_device))[0]

    all_text_embeddings = torch.cat([negative_text_embeddings, positive_text_embeddings]).detach().cpu().numpy()

    unconditional_embeddings = all_text_embeddings[0]
    conditional_embeddings = all_text_embeddings[1]
    embeddings_shape = list(unconditional_embeddings.shape)
    unconditional_embeddings = unconditional_embeddings.flatten()
    conditional_embeddings = conditional_embeddings.flatten()

    embeddings = {}
    embeddings["unconditional_embedding"] = unconditional_embeddings
    embeddings["conditional_embedding"] = conditional_embeddings
    embeddings["tensor_shape"] = embeddings_shape
    return embeddings
