import logging

import numpy
import torch
from compel import Compel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline
logging.disable(logging.WARNING)


def run(
    input_prompt_positive,
    input_prompt_negative,
    torch_device,
    model,
    local_cache_only=True,
):
    # try:
    #     text_encoder = CLIPTextModel.from_pretrained(
    #         model, subfolder="text_encoder", local_files_only=local_cache_only
    #     )
    # except OSError:
    #     text_encoder = CLIPTextModel.from_pretrained(
    #         model, local_files_only=local_cache_only
    #     )
    # except Exception as err:
    #     print(f"Unexpected {err}, {type(err)}")
    # text_encoder.to(torch_device)

    # try:
    #     tokenizer = CLIPTokenizer.from_pretrained(
    #         model, subfolder="tokenizer", local_files_only=local_cache_only
    #     )
    # except OSError as error:
    #     tokenizer = CLIPTokenizer.from_pretrained(
    #         model, local_files_only=local_cache_only
    #     )
    # except Exception as err:bbbb
    #     print(f"Unexpected {err}, {type(err)}")

    def unpack_values(data):
        a, *remainder = data
        b = remainder[0] if remainder else a
        return a, b

    pipeline = DiffusionPipeline.from_pretrained(model, local_files_only=local_cache_only)
    XL = False
    try:
        compel_proc = Compel(requires_pooled=[False, True], tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2], text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2])
        XL = True
    except:
        compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

    conditional_embeddings, conditional_embeddings_pooled = unpack_values(compel_proc(input_prompt_positive))
    unconditional_embeddings, unconditional_embeddings_pooled = unpack_values(compel_proc(input_prompt_negative))

    embeddings_shape = list(unconditional_embeddings.shape)
    if not XL:
        embeddings_shape.insert(0, 0)

    unconditional_embeddings = unconditional_embeddings.flatten()
    unconditional_embeddings_pooled = unconditional_embeddings_pooled.flatten()
    conditional_embeddings = conditional_embeddings.flatten()
    conditional_embeddings_pooled = conditional_embeddings_pooled.flatten()

    embeddings = {}
    embeddings["unconditional_embedding"] = unconditional_embeddings
    embeddings["conditional_embedding"] = conditional_embeddings
    embeddings["unconditional_embedding_pooled"] = unconditional_embeddings_pooled
    embeddings["conditional_embedding_pooled"] = conditional_embeddings_pooled
    embeddings["tensor_shape"] = embeddings_shape
    return embeddings
