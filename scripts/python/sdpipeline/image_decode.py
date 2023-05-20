import numpy
import torch
from diffusers import AutoencoderKL


def run(
    input_latents,
    latent_dimension,
    torch_device,
    model,
    local_cache_only=True,
    seamless_gen=False,
):
    vae = AutoencoderKL.from_pretrained(
        model, subfolder="vae", local_files_only=local_cache_only
    )
    vae.to(torch_device)

    if seamless_gen:
        for module in vae.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = "circular"

    latents = torch.from_numpy(
        numpy.array(
            [input_latents.reshape(4, latent_dimension[1], latent_dimension[0])]
        )
    ).to(torch_device)
    latents = (1.0 / vae.config.scaling_factor) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    return image.cpu().numpy()[0]
