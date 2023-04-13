from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from . import schedulers_lookup
from diffusers import ControlNetModel
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

    #guided = t_start >= 0
    # masking = len(mask_latents) > 0 and guided

    mask = torch.from_numpy(numpy.array([numpy.array([numpy.array(mask_latents.reshape(4, latent_dimension[0], latent_dimension[1]))[0]])]))
    mask = torch.cat([mask] * 2).to(torch_device)


    masked_image = torch.from_numpy(numpy.array([numpy.array(input_scheduler["masked_image_latent"].reshape(4, latent_dimension[0], latent_dimension[1]))]))
    masked_image = torch.cat([masked_image] * 2).to(torch_device)


    #mask = torch.from_numpy(numpy.array(mask_latents.reshape(4, latent_dimension[0], latent_dimension[1]))[0]).to(torch_device) 
   # noise = torch.from_numpy(numpy.array([input_scheduler["noise_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)

    # if not guided:
    #     # Only text prompt, starting with just noise latents
    #     init_latents = noise
    #     init_latents = init_latents * scheduler.init_noise_sigma
    # elif guided and not masking:# if GUIDED_ONLY:
    #     # Fed by guided image, so starting with image latents + noise
    #     init_latents = torch.from_numpy(numpy.array([input_scheduler["guided_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    # elif guided and masking:
        # Fed by guided image, so starting with image latents + noise, but only for masked regions
    # init_latents_orig = torch.from_numpy(numpy.array([input_scheduler["image_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    init_latents = torch.from_numpy(numpy.array([input_scheduler["guided_latent"].reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)

    # else:
    #     raise hou.nodeWarning("Incorrect input data! Please provide guide image, or guide image and mask!")

    # CONTROLNET #
    

    controlnetmodel = ""
    controlnet_image = []
    for prim in controlnet_geo.points()[:1]:
        controlnetmodel = prim.stringAttribValue("model")
        geo = prim.prims()[0].getEmbeddedGeometry()
        floats = numpy.array(geo.pointFloatAttribValues("Cd"))
        width = geo.attribValue("width")
        height = geo.attribValue("height")
        controlnet_conditioning_image = torch.from_numpy(numpy.array([floats.reshape(3, width, height)])).to(device=torch_device)
        controlnet_conditioning_image = controlnet_conditioning_image.to(torch.float16)


    controlnet = ControlNetModel.from_pretrained(controlnetmodel, local_files_only=local_cache_only, torch_dtype=torch.float16)
    controlnet.to(torch_device)
    controlnet_conditioning_scale = 1.0
    # CONTROLNET #


    text_embeddings = numpy.array(input_embeddings["conditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    uncond_embeddings = numpy.array(input_embeddings["unconditional_embedding"]).reshape(input_embeddings["tensor_shape"])
    text_embeddings = torch.from_numpy(numpy.array([uncond_embeddings, text_embeddings])).to(torch_device).half()

    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    latents = init_latents

    with hou.InterruptableOperation("Solving Stable Diffusion", open_interrupt_dialog=True) as operation:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t).to(torch.float16)      


            # CONTROLNET
            inpainting_latent_model_input = torch.cat([latent_model_input, mask, masked_image], dim=1).to(torch.float16)

            down_block_res_samples, mid_block_res_sample = controlnet(latent_model_input, t.to(torch.float16), encoder_hidden_states=text_embeddings, controlnet_cond=controlnet_conditioning_image, return_dict=False,)
            down_block_res_samples = [(down_block_res_sample * controlnet_conditioning_scale).to(torch.float16) for down_block_res_sample in down_block_res_samples]
            mid_block_res_sample *= controlnet_conditioning_scale


            with torch.no_grad():
                noise_pred = unet(inpainting_latent_model_input, t.to(torch.float16), encoder_hidden_states=text_embeddings, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, ).sample
            # CONTROLNET


            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # if masking:
            #     init_latents_proper = scheduler.add_noise(init_latents_orig, noise, t)
            #     latents = (init_latents_proper * (1.0-mask)) + (latents * mask)

            operation.updateProgress(i/len(timesteps))

    # # if guided and masking:
    # #     latents = (init_latents_orig * (1.0-mask)) + (latents * mask)

    latents = latents.cpu().numpy()[0]

    return latents
