from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from . import schedulers_lookup
import torch
import json
import numpy

def run(iference_steps, latent_dimension, input_embeddings, cfg_scale, input_scheduler, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", local_files_only=local_cache_only)
    unet.to(torch_device)

    guidance_scale = cfg_scale                # Scale for classifier-free guidance
    ATTR_UNET_LATENTS = torch.from_numpy(numpy.array([input_scheduler["latents"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device) 
    _unconditional_embedding = numpy.array(input_embeddings["unconditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    _conditional_embedding = numpy.array(input_embeddings["conditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    _text_embedding = torch.from_numpy(numpy.array([_unconditional_embedding, _conditional_embedding])).to(torch_device)

    scheduler_config = input_scheduler["config"]
    init_timesteps = scheduler_config["init_timesteps"]
    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]

    __scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(scheduler_config)
    __scheduler.set_timesteps(iference_steps)

    
    timesteps = __scheduler.timesteps
    if init_timesteps >= 0:
        timesteps = __scheduler.timesteps[init_timesteps:].to(torch_device)

    for t in tqdm(timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([ATTR_UNET_LATENTS] * 2)
        latent_model_input = __scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=_text_embedding).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        ATTR_UNET_LATENTS = __scheduler.step(noise_pred, t, ATTR_UNET_LATENTS).prev_sample
    

    ATTR_UNET_OUT_LATENTS = ATTR_UNET_LATENTS.cpu().numpy()[0]
    return ATTR_UNET_OUT_LATENTS
    ##### UNET SOLVER NODE ########

