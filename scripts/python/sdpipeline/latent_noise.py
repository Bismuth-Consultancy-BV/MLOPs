from diffusers import UNet2DConditionModel
import torch

# TODO: FIX THE DEPENDENCY FOR THE unet.in_channels
def run(text_prompt, noise_seed, width, height, torch_device):
	_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", local_files_only=True)
	_unet.to(torch_device)

	generator = torch.manual_seed(noise_seed)    # Seed generator to create the inital latent noise
	latents = torch.randn((len(text_prompt), _unet.in_channels, height // 8, width // 8),generator=generator,)
	latents = latents.to(torch_device)
	ATTR_LATENTS = latents.cpu().numpy()[0].flatten()
	return ATTR_LATENTS
