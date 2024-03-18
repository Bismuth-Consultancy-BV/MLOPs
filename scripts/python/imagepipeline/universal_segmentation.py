import mlops_image_utils
import numpy
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import os

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def run(model, image_colors, torch_device='cuda', cache_only=False):
    image = mlops_image_utils.colors_numpy_array_to_pil(image_colors).convert("RGB")
        
    if torch_device.startswith("cuda"):
        torch.cuda.empty_cache()

    former_model = AutoModelForUniversalSegmentation.from_pretrained(
        model, local_files_only=cache_only
    ).to(torch_device)

    processor = AutoProcessor.from_pretrained(
        model, 
        local_files_only=cache_only
    )
    inputs = processor(
        image, 
        task_inputs=['panoptic'],
        return_tensors="pt"
    ).to(torch_device)
    
    with torch.no_grad():
        outputs = former_model(**inputs)

    labels = former_model.config.id2label
    masks = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.size[1],image.size[0])])
    mask = masks[0]
        
    if torch_device.startswith("cuda"):
        torch.cuda.empty_cache()

    return mask, labels

def load_image(url):
    """Load an image from a URL or a local file path."""
    if url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    elif url.startswith('file://'):
        file_path = url.replace('file://', '')
        image = Image.open(file_path)
    else:
        # Assuming it's a direct file path without 'file://' scheme
        image = Image.open(url)
    
    return image
