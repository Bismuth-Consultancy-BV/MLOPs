import diffusers
from imp import reload
reload(diffusers)

import mlops_utils
import mlops_image_utils
import torch
from . import schedulers_lookup
import numpy
import os

def run(
    model,
    cache_only,
    inference_steps,
    guidance_scale,
    image_deviation,
    input_scheduler,
    input_embeddings,
    controlnet_geo,
):

    dtype = torch.float16

    input_controlnet_models = []
    input_controlnet_images = []
    input_controlnet_scales = []

    if controlnet_geo:
        for point in controlnet_geo.points():
            controlnetmodel = point.stringAttribValue("model")
            geo = point.prims()[0].getEmbeddedGeometry()
            controlnet_conditioning_scale = point.attribValue("scale")
            input_colors = mlops_image_utils.colored_points_to_numpy_array(geo)

            controlnet_conditioning_image = torch.from_numpy(
                numpy.array([input_colors])
            ).to(dtype)

            controlnet = diffusers.ControlNetModel.from_pretrained(
                controlnetmodel,
                local_files_only=cache_only,
                torch_dtype=dtype,
            )

            input_controlnet_models.append(controlnet)
            input_controlnet_images.append(controlnet_conditioning_image)
            input_controlnet_scales.append(controlnet_conditioning_scale)

    # Text Embeddings
    conditional_embeddings = torch.from_numpy(numpy.array([numpy.array(input_embeddings["conditional_embedding"]).reshape(
        input_embeddings["tensor_shape"])]))

    unconditional_embeddings = torch.from_numpy(numpy.array([numpy.array(input_embeddings["unconditional_embedding"]).reshape(
        input_embeddings["tensor_shape"])]))

    # Guide Image
    input_image = mlops_image_utils.colors_numpy_array_to_pil(input_scheduler["numpy_image"])

    # Mask
    input_mask = mlops_image_utils.colors_numpy_array_to_pil(input_scheduler["numpy_mask"])

    # Model
    model_path = mlops_utils.ensure_huggingface_model_local(model, os.path.join("$MLOPS", "data", "models", "diffusers"), cache_only)

    # Diffusers Pipeline
    pipe = diffusers.StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_path,  controlnet=input_controlnet_models, torch_dtype=dtype)

    # Scheduler
    scheduler_config = input_scheduler["config"]
    seed = input_scheduler["seed"]
    scheduler_type = scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    # Inference
    pipe.enable_model_cpu_offload()
    output_image = pipe(
        prompt_embeds = conditional_embeddings,
        negative_prompt_embeds = unconditional_embeddings,
        num_inference_steps=inference_steps,
        guidance_scale = guidance_scale,
        eta=1.0,
        image=input_image,
        mask_image=input_mask,
        control_image=input_controlnet_images,
        controlnet_conditioning_scale=input_controlnet_scales,
        output_type = "pil",
        generator = torch.manual_seed(seed),
        # TODO: FIX the clamping below. Should not be there, but the solve throws an error otherwise
        strength= max(0.05, image_deviation)
    ).images[0]

    return output_image
