import json

import numpy
import torch

from . import schedulers_lookup


def run(
    input_latents,
    latent_dimension,
    image_latents,
    guiding_strength,
    inference_steps,
    torch_device,
    scheduler_model,
    model,
    local_cache_only=True,
):
    try:
        scheduler_object = schedulers_lookup.schedulers[
            scheduler_model
        ].from_pretrained(
            model, subfolder="scheduler", local_files_only=local_cache_only
        )
    except OSError:
        scheduler_object = schedulers_lookup.schedulers[
            scheduler_model
        ].from_pretrained(model, local_files_only=local_cache_only)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")

    scheduler_object.set_timesteps(inference_steps)
    noise_latents = torch.from_numpy(
        numpy.array(
            [input_latents.reshape(4, latent_dimension[1], latent_dimension[0])]
        )
    )

    # If torch_device is `mps``, make sure dtype is set to float32
    # as currently MPS cannot handle float64
    if torch_device == "mps":
        noise_latents = noise_latents.to(torch.float32)
    noise_latents = noise_latents.to(torch_device)

    scheduler = {}
    scheduler["guided_latents"] = noise_latents.cpu().numpy()[0]

    # Figuring initial time step based on strength
    init_timestep = int(inference_steps * guiding_strength)
    init_timestep = min(init_timestep, inference_steps)

    timesteps = scheduler_object.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device=torch_device)
    t_start = 0

    if len(image_latents) != 0:
        if guiding_strength > 0.05 and guiding_strength < 1.0:
            image_latents = torch.from_numpy(
                numpy.array(
                    [image_latents.reshape(4, latent_dimension[1], latent_dimension[0])]
                )
            )
            # for mps device, make sure dtype is set to float32
            if torch_device == "mps":
                image_latents = image_latents.to(torch.float32)
            image_latents = image_latents.to(torch_device)
            guided_latents = scheduler_object.add_noise(
                image_latents, noise_latents, timesteps
            )
            scheduler["guided_latents"] = guided_latents.cpu().numpy()[0]
            t_start = max(inference_steps - init_timestep, 0)

    config = scheduler_object.config
    config["init_timesteps"] = t_start
    config["type"] = scheduler_model
    # A bit of a lame way to add Karras sigmas : just check if the menu label contains "Karras". bandaid.
    if "karras" in str(scheduler_model).lower():
        config["use_karras_sigmas"] = True
    scheduler["config"] = config
    return scheduler


def run_new(
    input_latents,
    latent_dimension,
    image_latents,
    guiding_strength,
    inference_steps,
    torch_device,
    scheduler_model,
    model,
    local_cache_only=True,
    do_half=True,    
    do_defer=False,
    do_exact=False,
):
    do_guide = image_latents is not None    
    dtype_torch = torch.float16 if do_half else torch.float32
    dtype_numpy = numpy.float16 if do_half else numpy.float32
    # setup scheduler
    try:
        scheduler_object = schedulers_lookup.schedulers[
            scheduler_model
        ].from_pretrained(
            model, subfolder="scheduler", local_files_only=local_cache_only, torch_dtype=dtype_torch
        )
    except OSError:
        scheduler_object = schedulers_lookup.schedulers[
            scheduler_model
        ].from_pretrained(model, local_files_only=local_cache_only, torch_dtype=dtype_torch)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")

    scheduler_object.set_timesteps(inference_steps)
    

    scheduler = {}    
    scheduler["guided_latents"] = input_latents

    # Figuring initial time step based on strength
    init_timestep = int(inference_steps * guiding_strength)
    init_timestep = min(init_timestep, inference_steps)
    if do_exact:
        pass
        #init_timestep = inference_steps
    t_start = 0

    if do_guide:
        if guiding_strength > 0.05 and guiding_strength < 1.0: # why 0.05 lower limit?
            t_start = max(inference_steps - init_timestep, 0)
            if not do_defer: # do modification now
                timesteps = scheduler_object.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps], device=torch_device)                
                shape = (4, latent_dimension[1], latent_dimension[0])    
                noise_latents = torch.from_numpy(input_latents).reshape(shape).to(torch_device)
                image_latents = torch.from_numpy(image_latents).reshape(shape).to(torch_device) # squeeze?            
                with torch.no_grad():                    
                    # add noise to our guide image based on timesteps                    
                    noise_latents = scheduler_object.add_noise(image_latents, noise_latents, timesteps)
                scheduler["guided_latents"] = noise_latents.detach().cpu().squeeze().numpy()

    config = scheduler_object.config
    config["init_timesteps"] = t_start
    config["type"] = scheduler_model

    # A bit of a lame way to add Karras sigmas : just check if the menu label contains "Karras". bandaid.
    if "karras" in str(scheduler_model).lower():
        config["use_karras_sigmas"] = True

    scheduler["config"] = config
    return scheduler
