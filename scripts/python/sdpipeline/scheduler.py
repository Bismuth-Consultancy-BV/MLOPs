from diffusers import EulerDiscreteScheduler, LMSDiscreteScheduler
import json
import torch
import numpy

def run(input_latents, inference_steps, torch_device, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
	#scheduler_object = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
	scheduler_object = LMSDiscreteScheduler.from_pretrained(model, subfolder="scheduler", local_files_only=local_cache_only)
	scheduler_object.set_timesteps(inference_steps)
	__LATENTS = torch.from_numpy(numpy.array([input_latents.reshape(4, 96, 96)])).to(torch_device)
	ATTR_SCHEDULER_LATENTS = __LATENTS * scheduler_object.init_noise_sigma
	scheduler = {}
	scheduler["config"] = scheduler_object.config
	scheduler["type"] = "LMSDiscreteScheduler"
	scheduler["latents"] = ATTR_SCHEDULER_LATENTS.cpu().numpy()[0]
	return scheduler

