import requests
import mlops_image_utils
from transformers import BlipProcessor, BlipForConditionalGeneration

def run(model, conditional_prompt, skip_special, input_colors, torch_device, local_cache_only):
    processor = BlipProcessor.from_pretrained(model, local_files_only=local_cache_only)
    model = BlipForConditionalGeneration.from_pretrained(model, local_files_only=local_cache_only).to(torch_device)
    image = mlops_image_utils.colors_numpy_array_to_pil(input_colors)

    if conditional_prompt:
        inputs = processor(image, conditional_prompt, return_tensors="pt").to(torch_device)
    else:
        inputs = processor(image, return_tensors="pt").to(torch_device)

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=skip_special)