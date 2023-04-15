import json
import torch
import numpy
from . import schedulers_lookup


def run(input_latents, latent_dimension, image_latents, guiding_strength, inference_steps, torch_device, scheduler_model, model, local_cache_only=True):
    try:
        scheduler_object = schedulers_lookup.schedulers[scheduler_model].from_pretrained(model, subfolder="scheduler", local_files_only=local_cache_only)
    except OSError:
        scheduler_object = schedulers_lookup.schedulers[scheduler_model].from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")

    scheduler_object.set_timesteps(inference_steps)
    noise_latents = torch.from_numpy(numpy.array([input_latents.reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)

    scheduler = {}
    scheduler["guided_latents"] = noise_latents.cpu().numpy()[0]

    # Figuring initial time step based on strength
    init_timestep = int(inference_steps * guiding_strength)
    init_timestep = min(init_timestep, inference_steps)

    timesteps = scheduler_object.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device=torch_device)

    if len(image_latents) != 0:
        image_latents = torch.from_numpy(numpy.array([image_latents.reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
        guided_latents = scheduler_object.add_noise(image_latents, noise_latents, timesteps)
        scheduler["guided_latents"] = guided_latents.cpu().numpy()[0]
    t_start = max(inference_steps - init_timestep, 0)

    config = scheduler_object.config
    config["init_timesteps"] = t_start
    config["type"] = scheduler_model
    scheduler["config"] = config
    return scheduler
