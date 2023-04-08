from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from . import schedulers_lookup
import torch
import json
import numpy
import hou

def run(inference_steps, latent_dimension, input_embeddings, mask_latents, attention_slicing, guidance_scale, input_scheduler, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", local_files_only=local_cache_only)
    unet.to(torch_device)

    if attention_slicing:
        unet.set_attention_slice("auto")

    ATTR_UNET_LATENTS = torch.from_numpy(numpy.array([input_scheduler["latents"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device) 
    text_embeddings = numpy.array(input_embeddings["conditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    uncond_embeddings = numpy.array(input_embeddings["unconditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    
    text_embeddings = torch.from_numpy(numpy.array([uncond_embeddings, text_embeddings])).to(torch_device)

    scheduler_config = input_scheduler["config"]
    init_timesteps = scheduler_config["init_timesteps"]
    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]

    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(scheduler_config)
    scheduler.set_timesteps(inference_steps)

    
    timesteps = scheduler.timesteps
    if init_timesteps >= 0:
        timesteps = scheduler.timesteps[init_timesteps:].to(torch_device)

    latents = ATTR_UNET_LATENTS
    with hou.InterruptableOperation("Solving Stable Diffusion", open_interrupt_dialog=True) as operation:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            operation.updateProgress(i/len(timesteps))
    

    ATTR_UNET_OUT_LATENTS = latents.cpu().numpy()[0]
    return ATTR_UNET_OUT_LATENTS
    ##### UNET SOLVER NODE ########