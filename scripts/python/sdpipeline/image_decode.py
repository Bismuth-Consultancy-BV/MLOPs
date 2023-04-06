from diffusers import AutoencoderKL
import torch
import numpy

def run(input_latents, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
    # scale and decode the image latents with vae
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 

    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_cache_only)
    vae.to(torch_device)

    __LATENTS = torch.from_numpy(numpy.array([input_latents.reshape(4, 96, 96)])).to(torch_device)
    __LATENTS = 1 / 0.18215 * __LATENTS
    with torch.no_grad():
        image = vae.decode(__LATENTS).sample
    return image.cpu().numpy()[0]
    ######## IMAGE DECODER NODE ########