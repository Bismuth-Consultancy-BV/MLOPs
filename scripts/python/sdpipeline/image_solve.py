import hou
import mlops_image_utils
import numpy
import torch
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from tqdm.auto import tqdm

from . import schedulers_lookup


def run(
    inference_steps,
    latent_dimension,
    input_embeddings,
    controlnet_geo,
    attention_slicing,
    guidance_scale,
    input_scheduler,
    torch_device,
    model,
    lora,
    local_cache_only=True,
    seamless_gen=False,
):
    no_half = torch_device in ["mps", "cpu"]
    dtype_unet = torch.float32 if no_half else torch.float16
    dtype_controlnet = numpy.float32 if no_half else numpy.float16
    scheduler_config = input_scheduler["config"]
    t_start = scheduler_config["init_timesteps"]

    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(
        scheduler_config
    )
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps

    cross_attention_kwargs = {}
    unet = UNet2DConditionModel.from_pretrained(
        model,
        subfolder="unet",
        local_files_only=local_cache_only,
        torch_dtype=dtype_unet,
    )
    if lora["weights"] != "":
        cross_attention_kwargs = {"scale": lora["scale"]}
        unet.load_attn_procs(
            lora["weights"], use_safetensors=lora["weights"].endswith(".safetensors")
        )
    unet.to(torch_device)

    if seamless_gen:
        for module in unet.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = "circular"

    if attention_slicing:
        unet.set_attention_slice("auto")

    mask_latents = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["mask_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    mask_orig = mask_latents.to(torch_device)

    init_latents_orig = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["image_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    init_latents = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["guided_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    noise = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["noise_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    masking = input_scheduler["image_provided"] == 1

    text_embeddings = numpy.array(input_embeddings["conditional_embedding"]).reshape(
        input_embeddings["tensor_shape"]
    )
    uncond_embeddings = numpy.array(
        input_embeddings["unconditional_embedding"]
    ).reshape(input_embeddings["tensor_shape"])
    text_embeddings = torch.from_numpy(
        numpy.array([uncond_embeddings, text_embeddings])
    ).to(torch_device)

    if not no_half:
        text_embeddings = text_embeddings.half()

    if controlnet_geo:
        controlnet_model = []
        controlnet_image = []
        controlnet_scale = []
        for point in controlnet_geo.points():
            controlnetmodel = point.stringAttribValue("model")
            geo = point.prims()[0].getEmbeddedGeometry()
            controlnet_conditioning_scale = point.attribValue("scale")
            input_colors = mlops_image_utils.colored_points_to_numpy_array(geo)

            controlnet_conditioning_image = torch.from_numpy(
                numpy.array([input_colors])
            ).to(device=torch_device)
            controlnet_conditioning_image = controlnet_conditioning_image.to(dtype_unet)
            controlnet = ControlNetModel.from_pretrained(
                controlnetmodel,
                local_files_only=local_cache_only,
                torch_dtype=dtype_unet,
            )
            controlnet.to(torch_device)
            controlnet_model.append(controlnet)
            controlnet_image.append(controlnet_conditioning_image)
            controlnet_scale.append(controlnet_conditioning_scale)

        controlnet = MultiControlNetModel(controlnet_model)

        if seamless_gen:
            for module in controlnet.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.padding_mode = "circular"

    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    latents = init_latents

    with hou.InterruptableOperation(
        "Solving Stable Diffusion", open_interrupt_dialog=True
    ) as operation:
        # if True:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            latent_model_input = torch.cat([latents] * 2)
            # converting timestep to cpu for compatibility with all samplers
            t = t.to("cpu")
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            ).to(dtype_unet)

            if controlnet_geo:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t.to(dtype_unet),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                )

                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t.to(dtype_unet),
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t.to(dtype_unet),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if masking:
                init_latents_proper = scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * (1.0 - mask_orig)) + (
                    latents * mask_orig
                )

            operation.updateProgress(i / len(timesteps))

    if masking:
        latents = (init_latents_orig * (1.0 - mask_orig)) + (latents * mask_orig)
    latents = latents.cpu().numpy()[0]

    return latents
