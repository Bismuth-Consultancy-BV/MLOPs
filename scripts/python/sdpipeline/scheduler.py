from . import schedulers_lookup
import json
import torch
import numpy

def run(input_latents, latent_dimension, guided_latents, guiding_strength, guiding_seed, inference_steps, torch_device, scheduler_model, model="CompVis/stable-diffusion-v1-4", local_cache_only=True):
    try:
        scheduler_object = schedulers_lookup.schedulers[scheduler_model].from_pretrained(model, subfolder="scheduler", local_files_only=local_cache_only)
    except OSError:
        scheduler_object = schedulers_lookup.schedulers[scheduler_model].from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    scheduler_object.set_timesteps(inference_steps)
    __LATENTS = torch.from_numpy(numpy.array([input_latents.reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
    
    scheduler = {}
    t_start = -1

    if len(guided_latents) == 0:
         #, dtype=numpy.float64
        ATTR_SCHEDULER_LATENTS = __LATENTS * scheduler_object.init_noise_sigma
    else:
        __GUIDEDLATENTS = torch.from_numpy(numpy.array([guided_latents.reshape(4, latent_dimension[0], latent_dimension[1])])).to(torch_device)
        # Figuring initial time step based on strength
        init_timestep = int(inference_steps * guiding_strength) 

        timesteps = scheduler_object.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps], device=torch_device)
        
        ATTR_SCHEDULER_LATENTS = scheduler_object.add_noise(__GUIDEDLATENTS, __LATENTS, timesteps)
        t_start = max(inference_steps - init_timestep, 0)

    config = scheduler_object.config
    config["init_timesteps"] = t_start
    config["type"] = scheduler_model
    scheduler["config"] = config
    # scheduler["type"] = scheduler_model
    scheduler["latents"] = ATTR_SCHEDULER_LATENTS.cpu().numpy()[0]
    return scheduler