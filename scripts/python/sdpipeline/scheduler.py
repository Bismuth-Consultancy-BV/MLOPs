from diffusers import EulerDiscreteScheduler, LMSDiscreteScheduler
import json
import torch
import numpy

def run(input_latents, torch_device):
	scheduler_object = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
	__LATENTS = torch.from_numpy(numpy.array([input_latents.reshape(4, 64, 64)])).to(torch_device)
	ATTR_SCHEDULER_LATENTS = __LATENTS * scheduler_object.init_noise_sigma
	scheduler = {}
	scheduler["config"] = scheduler_object.config
	scheduler["type"] = "LMSDiscreteScheduler"
	scheduler["latents"] = ATTR_SCHEDULER_LATENTS.cpu().numpy()[0]
	return scheduler

