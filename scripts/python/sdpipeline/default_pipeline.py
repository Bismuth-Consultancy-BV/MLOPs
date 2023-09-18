import diffusers
from importlib import reload
reload(diffusers)


import mlops_utils
reload(mlops_utils)
import mlops_image_utils
import torch
from . import schedulers_lookup
from . import pipelines_lookup
reload(pipelines_lookup)
import numpy
import os
import hou

def run(
    model,
    cache_only,
    device,
    pipeline,
    resolution,
    inference_steps,
    guidance_scale,
    image_deviation,
    input_scheduler,
    input_embeddings,
    controlnet_geo,
    lora_weights,
):
    diffusers.utils.logging.set_verbosity_error()
    
    diffusers.utils.logging.set_verbosity(40)
    pipeline_call_kwargs = {}
    pipeline_kwargs = {}
    dtype = torch.float16

    pipeline_modifier = ""
    if pipeline == "stablediffusionxl":
        pipeline_modifier = "XL"

    pipeline_type = f"StableDiffusion{pipeline_modifier}Pipeline"

    if input_scheduler["numpy_image"] is not None:
        # Guide Image
        input_image = mlops_image_utils.colors_numpy_array_to_pil(input_scheduler["numpy_image"])
        pipeline_call_kwargs["image"] = input_image
        # Mask
        input_mask = mlops_image_utils.colors_numpy_array_to_pil(input_scheduler["numpy_mask"])
        pipeline_call_kwargs["mask_image"] = input_mask

        pipeline_call_kwargs["strength"] = max(0.05, image_deviation)
        pipeline_type = f"StableDiffusion{pipeline_modifier}InpaintPipeline"

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

        pipeline_call_kwargs["control_image"] = input_controlnet_images
        pipeline_call_kwargs["controlnet_conditioning_scale"] = input_controlnet_scales
        pipeline_kwargs["controlnet"] = input_controlnet_models
        pipeline_type = f"StableDiffusion{pipeline_modifier}ControlNetInpaintPipeline"

    # Text Embeddings
    tensor_shape = input_embeddings["tensor_shape"]
    if tensor_shape[0] == -1:
        tensor_shape = tensor_shape[1:]

    pooled_tensor_shape = input_embeddings["pooled_tensor_shape"]

    conditional_embeddings = torch.from_numpy(numpy.array(numpy.array(input_embeddings["conditional_embedding"]).reshape(
        tensor_shape)))

    pooled_conditional_embeddings = input_embeddings["pooled_conditional_embedding"][:pooled_tensor_shape[0]*pooled_tensor_shape[1]]
    pooled_conditional_embeddings = torch.from_numpy(numpy.array(numpy.array(pooled_conditional_embeddings).reshape(
        pooled_tensor_shape)))

    unconditional_embeddings = torch.from_numpy(numpy.array(numpy.array(input_embeddings["unconditional_embedding"]).reshape(
        tensor_shape)))
    pooled_unconditional_embeddings = input_embeddings["pooled_unconditional_embedding"][:pooled_tensor_shape[0]*pooled_tensor_shape[1]]
    pooled_unconditional_embeddings = torch.from_numpy(numpy.array(numpy.array(pooled_unconditional_embeddings).reshape(
        pooled_tensor_shape)))

    if pipeline == "stablediffusionxl":
        pipeline_call_kwargs["pooled_prompt_embeds"] = pooled_conditional_embeddings
        pipeline_call_kwargs["negative_pooled_prompt_embeds"] = pooled_unconditional_embeddings

    # Model
    model_path = mlops_utils.ensure_huggingface_model_local(model, os.path.join("$MLOPS", "data", "models", "diffusers"), cache_only)

    # Diffusers Pipeline
    pipe = pipelines_lookup.pipelines[pipeline_type].from_pretrained(model_path, torch_dtype=dtype,use_safetensors=True, **pipeline_kwargs)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Scheduler
    scheduler_config = input_scheduler["config"]
    seed = input_scheduler["seed"]
    scheduler_type = scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    # LORA
    lora_kwargs = {}
    if lora_weights["model"] != "":
        lora_model_path = mlops_utils.ensure_huggingface_model_local(lora_weights["model"], os.path.join("$MLOPS", "data", "models", "diffusers"), cache_only,model_type="all")
        pipe.load_lora_weights(lora_model_path)
        lora_kwargs = {"scale": lora_weights["weight"]}

    from functools import partial
    total_steps = min(int(inference_steps * image_deviation), inference_steps)
    def progress_bar(step, timestep, latents, operation):
        operation.updateProgress(step / total_steps)

    # Inference
    pipe.enable_model_cpu_offload()

    with hou.InterruptableOperation("Solving Stable Diffusion", open_interrupt_dialog=True) as operation:
        _progress_bar = partial(progress_bar, operation=operation)
        output_image = pipe(
            prompt_embeds=conditional_embeddings,
            negative_prompt_embeds=unconditional_embeddings,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            eta=1.0,
            width=resolution[0],
            height=resolution[1],
            output_type="pil",
            generator=torch.manual_seed(seed),
            cross_attention_kwargs=lora_kwargs,
            callback=_progress_bar,
            **pipeline_call_kwargs,
        ).images[0]

    return output_image
