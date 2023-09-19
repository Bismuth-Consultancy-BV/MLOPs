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
from typing import Optional
import inspect

def run(
    model,
    cache_only,
    device,
    pipeline,
    resolution,
    inference_steps,
    guidance_scale,
    image_deviation,
    tiling,
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
    if pipeline["type"] == "stablediffusionxl":
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

    conditional_embeddings = torch.from_numpy(numpy.array(numpy.array(input_embeddings["conditional_embedding"]).reshape(
        tensor_shape)))

    unconditional_embeddings = torch.from_numpy(numpy.array(numpy.array(input_embeddings["unconditional_embedding"]).reshape(
        tensor_shape)))
    

    # if pipeline["type"] == "stablediffusionxl":
    pooled_tensor_shape = input_embeddings["pooled_tensor_shape"]
    pooled_conditional_embeddings = input_embeddings["pooled_conditional_embedding"][:pooled_tensor_shape[0]*pooled_tensor_shape[1]]
    pooled_conditional_embeddings = torch.from_numpy(numpy.array(numpy.array(pooled_conditional_embeddings).reshape(
    pooled_tensor_shape)))
    pooled_unconditional_embeddings = input_embeddings["pooled_unconditional_embedding"][:pooled_tensor_shape[0]*pooled_tensor_shape[1]]
    pooled_unconditional_embeddings = torch.from_numpy(numpy.array(numpy.array(pooled_unconditional_embeddings).reshape(
    pooled_tensor_shape)))

    pipeline_call_kwargs["pooled_prompt_embeds"] = pooled_conditional_embeddings
    pipeline_call_kwargs["negative_pooled_prompt_embeds"] = pooled_unconditional_embeddings

    del input_embeddings
    del controlnet_geo

    # Model
    model_path = mlops_utils.ensure_huggingface_model_local(model, os.path.join("$MLOPS", "data", "models", "diffusers"), cache_only)
    

    # Delete kwargs not used by pipeline
    _keep = inspect.signature(pipelines_lookup.pipelines[pipeline_type]).parameters.keys()
    for key, value in pipeline_kwargs.copy().items():
        if key not in _keep:
            del pipeline_kwargs[key]

    # Diffusers Pipeline
    if pipeline["type"] == "custom":
        pipeline_type = pipeline["name"]

    pipe = pipelines_lookup.pipelines[pipeline_type].from_pretrained(model_path, torch_dtype=dtype,use_safetensors=True, **pipeline_kwargs)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)


    # Tiling Support (Have to hijack pipeline to support this currently)
    if tiling != "none":
        if tiling == "x":
            modex = "circular"
            modey = "constant"
        if tiling == "y":
            modex = "constant"
            modey = "circular"
        if tiling == "xy":
            modex = "circular"
            modey = "circular"

        def asymmetricConv2DConvForward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
            F = torch.nn.functional
            self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
            self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
            working = F.pad(input, self.paddingX, mode=modex)
            working = F.pad(working, self.paddingY, mode=modey)
            return F.conv2d(working, weight, bias, self.stride, torch.nn.modules.utils._pair(0), self.dilation, self.groups)

        targets = [pipe.vae, pipe.text_encoder, pipe.unet,]
        conv_layers = []
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)

        for cl in conv_layers:
            if isinstance(cl, diffusers.models.lora.LoRACompatibleConv) and cl.lora_layer is None:
                cl.lora_layer = lambda *x: 0

            cl._conv_forward = asymmetricConv2DConvForward.__get__(cl, torch.nn.Conv2d)


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
    def progress_bar(step, timestep, latents, operation):
        operation.updateProgress(step / inference_steps)

    # Inference
    pipe.enable_model_cpu_offload()


    # Delete kwargs not used by pipeline
    _keep = inspect.signature(pipelines_lookup.pipelines[pipeline_type].__call__).parameters.keys()
    for key, value in pipeline_call_kwargs.copy().items():
        if key not in _keep:
            del pipeline_call_kwargs[key]


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
