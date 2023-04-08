import argparse
import inspect
import pathlib
from typing import List, Optional, Union

import numpy as np
import torch
from torch import autocast

import PIL
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def preprocess(image):
    image = image.convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


class StableDiffusionUnifiedPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        #scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Optional[PIL.Image.Image] = None,
        mask_image: Optional[PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 20,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
    ):

        batch_size = 1
        offset = 0

        self.scheduler.set_timesteps(num_inference_steps)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

        # preprocess image
        init_image = preprocess(init_image).to(self.device)

        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(init_image).latent_dist.sample()
        init_latents = 0.18215 * init_latents
        init_latents_orig = init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)


        # preprocess mask
        mask = preprocess_mask(mask_image).to(self.device)
        mask = torch.cat([mask] * batch_size)


        # get prompt text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer([""] * 1, padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        #### SOLVE ####
        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)

        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # masking
            if mask_image is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))











        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        #print(image)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return image


pipe = StableDiffusionUnifiedPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=False,
    local_files_only=False,
    device="cuda",
).to("cuda")


inpaint_image = PIL.Image.open(r"C:\Users\Paul\Downloads\mo.jpg").convert("RGB")
inpaint_image = PIL.ImageOps.exif_transpose(inpaint_image)
inpaint_image = inpaint_image.resize((512, 512))


inpaint_mask = PIL.Image.open(r"C:\Users\Paul\Downloads\mo_mask.jpg").convert("RGB")
inpaint_mask = PIL.ImageOps.exif_transpose(inpaint_mask)
inpaint_mask = inpaint_mask.resize((512, 512))


with autocast("cuda"):
    image = pipe(
        "beard",
        init_image=inpaint_image,
        mask_image=inpaint_mask,
        strength=0.8,
        guidance_scale=7.5,
        num_inference_steps=20,
    )[0]

    image.show()
