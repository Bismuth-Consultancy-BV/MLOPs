from transformers import CLIPTokenizer


def run(text_prompt, model, local_cache_only):
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_cache_only
        )
    except OSError as error:
        tokenizer = CLIPTokenizer.from_pretrained(
            model, local_files_only=local_cache_only
        )
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")

    text_input = tokenizer(
        text_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = list(text_input.input_ids.cpu().numpy()[0])
    return input_ids
