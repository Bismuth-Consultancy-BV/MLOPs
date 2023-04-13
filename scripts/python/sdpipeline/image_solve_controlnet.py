from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from . import schedulers_lookup
from diffusers import ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
import torch
import json
import numpy
import hou


def run(inference_steps, latent_dimension, input_embeddings, mask_latents, controlnet_geo, attention_slicing, guidance_scale, input_scheduler, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
    scheduler_config = input_scheduler["config"]
    t_start = scheduler_config["init_timesteps"]

    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(scheduler_config)
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", local_files_only=local_cache_only, torch_dtype=torch.float16)
    unet.to(torch_device)

    if attention_slicing:
        unet.set_attention_slice("auto")

    mask = torch.from_numpy(numpy.array([numpy.array([numpy.array(mask_latents.reshape(4, latent_dimension[0], latent_dimension[1]))[0]])]))
    mask_orig = mask.to(torch_device)
    mask = torch.cat([mask] * 2).to(torch_device)


    masked_image = torch.from_numpy(numpy.array([numpy.array(input_scheduler["masked_image_latent"].reshape(4, latent_dimension[0], latent_dimension[1]))]))
    masked_image = torch.cat([masked_image] * 2).to(torch_device)
    
    init_latents = torch.from_numpy(numpy.array([input_scheduler["guided_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    init_latents_orig = torch.from_numpy(numpy.array([input_scheduler["image_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    noise = torch.from_numpy(numpy.array([input_scheduler["noise_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)

    controlnet_model = []
    controlnet_image = []
    controlnet_scale = []
    for point in controlnet_geo.points():
        controlnetmodel = point.stringAttribValue("model")
        geo = point.prims()[0].getEmbeddedGeometry()
        floats = numpy.array(geo.pointFloatAttribValues("Cd"))
        width = geo.attribValue("width")
        height = geo.attribValue("height")
        controlnet_conditioning_scale = point.attribValue("scale")
        r = numpy.array(geo.pointFloatAttribValues("r"), dtype=numpy.float16).reshape(width, height)# * 2.0 - 1.0
        g = numpy.array(geo.pointFloatAttribValues("g"), dtype=numpy.float16).reshape(width, height)# * 2.0 - 1.0
        b = numpy.array(geo.pointFloatAttribValues("b"), dtype=numpy.float16).reshape(width, height)# * 2.0 - 1.0
        input_colors = numpy.stack((r,g,b), axis=0)
        controlnet_conditioning_image = torch.from_numpy(numpy.array([input_colors])).to(device=torch_device)
        controlnet_conditioning_image = controlnet_conditioning_image.to(torch.float16)
        controlnet = ControlNetModel.from_pretrained(controlnetmodel, local_files_only=local_cache_only, torch_dtype=torch.float16)
        controlnet.to(torch_device)
        controlnet_model.append(controlnet)
        controlnet_image.append(controlnet_conditioning_image)
        controlnet_scale.append(controlnet_conditioning_scale)

        # from PIL import Image
        # input_colors = input_colors*255.0
        # input_colors = input_colors.astype(numpy.uint8)
        # img = Image.fromarray(input_colors.transpose(1, 2, 0))
        # img.show()

    controlnet = MultiControlNetModel(controlnet_model)
    # controlnet = controlnet_model[0]
    # controlnet_image = controlnet_image[0]
    # controlnet_scale = controlnet_scale[0]

    text_embeddings = numpy.array(input_embeddings["conditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    uncond_embeddings = numpy.array(input_embeddings["unconditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    text_embeddings = torch.from_numpy(numpy.array([uncond_embeddings, text_embeddings])).to(torch_device).half()

   # print(timesteps)
    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    #print(timesteps)
    latents = init_latents

    with hou.InterruptableOperation("Solving Stable Diffusion", open_interrupt_dialog=True) as operation:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t).to(torch.float16)      


            # CONTROLNET
            inpainting_latent_model_input = torch.cat([latent_model_input, mask, masked_image], dim=1).to(torch.float16)
            down_block_res_samples, mid_block_res_sample = controlnet(latent_model_input, t.to(torch.float16), encoder_hidden_states=text_embeddings, controlnet_cond=controlnet_conditioning_image, conditioning_scale=controlnet_scale ,return_dict=False,)

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t.to(torch.float16), encoder_hidden_states=text_embeddings, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, ).sample
                # noise_pred = unet(inpainting_latent_model_input, t.to(torch.float16), encoder_hidden_states=text_embeddings, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # MASKING
            init_latents_proper = scheduler.add_noise(init_latents_orig, noise, t)
            latents = init_latents_proper * (1.0-mask_orig) + (latents * mask_orig)

            operation.updateProgress(i/len(timesteps))

    # FINAL MASKING
    latents = (init_latents_orig * (1.0-mask_orig)) + (latents * mask_orig)
    latents = latents.cpu().numpy()[0]

    return latents