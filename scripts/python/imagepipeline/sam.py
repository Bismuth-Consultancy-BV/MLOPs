import mlops_image_utils
import torch
import numpy
from transformers import pipeline, SamModel, SamProcessor

def run(model, input_colors, points, labels, torch_device, local_cache_only):

    image = mlops_image_utils.colors_numpy_array_to_pil(input_colors).convert("RGB")
    if torch_device == "cuda":
        torch.cuda.empty_cache()

    if len(points) == 0:
        if torch_device == "cuda":
            torch_device += ":0"

        generator = pipeline("mask-generation", model=model, device=torch_device)

        with torch.no_grad():
            outputs = generator(image, points_per_batch=64)

        masks = outputs["masks"]
        h,w = masks[0].shape[-2:]
        out = numpy.zeros((h,w,1))

        for i, mask in enumerate(masks):
            value = numpy.full((h,w, 1), i, dtype=numpy.uint8)
            mask = mask.reshape(h, w, 1)
            out += value * mask

    else:
        sam_model = SamModel.from_pretrained(model, local_files_only=local_cache_only).to(torch_device)
        processor = SamProcessor.from_pretrained(model, local_files_only=local_cache_only)
        inputs = processor(image, input_points=[points], input_labels=[labels], return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = sam_model(**inputs)

        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        mask = masks[0][0].numpy()
        mask = mask[0, :, :]
        dimension = mask.shape
        h = dimension[0]
        w = dimension[1]
        mask = mask.reshape(h, w, 1)
        out = numpy.zeros((h,w,1))
        value = numpy.full((h,w, 1), 1, dtype=numpy.uint8)
        out += value * mask

    if torch_device.startswith("cuda"):
        torch.cuda.empty_cache()

    return out