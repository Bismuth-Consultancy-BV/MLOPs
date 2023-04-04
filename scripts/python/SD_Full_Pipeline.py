######## TEXT PROMPT NODE ########
ATTR_PROMPT = ["a photograph of an astronaut riding a horse"]
######## TEXT PROMPT NODE ########





####### SYSTEM SETTINGS NODE #######
ATTR_TORCH_DEVICE = "cuda"
ATTR_IMAGE_HEIGHT = 512                        # default ATTR_IMAGE_HEIGHT of Stable Diffusion
ATTR_IMAGE_WIDTH = 512                         # default ATTR_IMAGE_WIDTH of Stable Diffusion
ATTR_NUM_INFERENCE_STEPS = 20           # Number of denoising steps
ATTR_SEED = 44
####### SYSTEM SETTINGS NODE #######






## DONE
###### TOKENIZER NODE ######
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
text_input = tokenizer(ATTR_PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
ATTR_INPUT_IDS = list(text_input.input_ids.cpu().numpy()[0])
###### TOKENIZER NODE ######








## TODO: text_embeddings
###### EMBEDDER NODE  ######
from transformers import CLIPTextModel
import torch
import numpy

__IDS = torch.from_numpy(numpy.array([ATTR_INPUT_IDS]))
#TODO: Remove duplicate tokenizer
batch_size = len(ATTR_PROMPT)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
text_encoder.to(ATTR_TORCH_DEVICE)
text_embeddings = text_encoder(__IDS.to(ATTR_TORCH_DEVICE))[0]
max_length = __IDS.shape[-1]
_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
uncond_input = _tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(ATTR_TORCH_DEVICE))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
del _tokenizer
###### EMBEDDER NODE  ######


###### LATENT NOISE NODE  ######
# TODO: FIX THE DEPENDENCY FOR THE unet.in_channels
from diffusers import UNet2DConditionModel
_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", local_files_only=True)
_unet.to(ATTR_TORCH_DEVICE)

import torch
batch_size = len(ATTR_PROMPT)
generator = torch.manual_seed(ATTR_SEED)    # Seed generator to create the inital latent noise
latents = torch.randn((batch_size, _unet.in_channels, ATTR_IMAGE_HEIGHT // 8, ATTR_IMAGE_WIDTH // 8),generator=generator,)
del _unet
latents = latents.to(ATTR_TORCH_DEVICE)
ATTR_LATENTS = latents.cpu().numpy()[0].flatten()
###### LATENT NOISE NODE  ######









###### SCHEDULER NODE #######
# Creating our pre-trained scheduler
from diffusers import LMSDiscreteScheduler
import json
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(ATTR_NUM_INFERENCE_STEPS)
__LATENTS = torch.from_numpy(numpy.array([ATTR_LATENTS.reshape(4, 64, 64)])).to(ATTR_TORCH_DEVICE)
ATTR_SCHEDULER_LATENTS = __LATENTS * scheduler.init_noise_sigma
ATTR_SCHEDULER_CONFIG = json.dumps(scheduler.config)
###### SCHEDULER NODE #######






##### UNET SOLVER NODE ########
# 3. The UNet model for generating the latents.
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", local_files_only=True)
unet.to(ATTR_TORCH_DEVICE)

from tqdm.auto import tqdm
import torch
import json
guidance_scale = 7.5                # Scale for classifier-free guidance
ATTR_UNET_LATENTS = ATTR_SCHEDULER_LATENTS
__scheduler = LMSDiscreteScheduler.from_config(json.loads(ATTR_SCHEDULER_CONFIG))
__scheduler.set_timesteps(ATTR_NUM_INFERENCE_STEPS)
for t in tqdm(__scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([ATTR_UNET_LATENTS] * 2)

    latent_model_input = __scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    ATTR_UNET_LATENTS = __scheduler.step(noise_pred, t, ATTR_UNET_LATENTS).prev_sample
##### UNET SOLVER NODE ########






######## IMAGE DECODER NODE ########
# scale and decode the image latents with vae
# 1. Load the autoencoder model which will be used to decode the latents into image space. 
from diffusers import AutoencoderKL
import torch
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", local_files_only=True)
vae.to(ATTR_TORCH_DEVICE)

ATTR_UNET_LATENTS = 1 / 0.18215 * ATTR_UNET_LATENTS
with torch.no_grad():
    image = vae.decode(ATTR_UNET_LATENTS).sample
######## IMAGE DECODER NODE ########






##### UTILITY TO WRITE IMAGE TO DISK #######
from PIL import Image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].show()
pil_images[0].save(r"C:\Users\Paul\OneDrive - Bismuth Consultancy B.V\BISMUTH_CONSULTANCY\Work\Projects\Monaco\Toolset\MLOPs\scripts\pythongeeks.jpg")
##### UTILITY TO WRITE IMAGE TO DISK #######
