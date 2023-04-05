from diffusers import UNet2DConditionModel
import torch

def run(noise_seed, width, height, torch_device):
	generator = torch.manual_seed(noise_seed)    # Seed generator to create the inital latent noise
	latents = torch.randn((1, 4, height // 8, width // 8),generator=generator,)
	latents = latents.to(torch_device)
	ATTR_LATENTS = latents.cpu().numpy()[0]
	return ATTR_LATENTS
