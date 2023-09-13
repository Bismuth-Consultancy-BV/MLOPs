import diffusers
from imp import reload
reload(diffusers)

# from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import mlops_utils
import mlops_image_utils
import torch
from . import schedulers_lookup
import numpy
import os

def run(
    model,
    cache_only,
    inference_steps,
    guidance_scale,
    image_deviation,
    input_scheduler,
    input_embeddings,
    controlnet_geo,
    latent_dimension,
):

    dtype = torch.float16
    
    controlnet_models = []
    controlnet_images = []
    controlnet_scales = []

    if controlnet_geo:
        for point in controlnet_geo.points():
            controlnetmodel = point.stringAttribValue("model")
            geo = point.prims()[0].getEmbeddedGeometry()
            controlnet_conditioning_scale = point.attribValue("scale")
            input_colors = mlops_image_utils.colored_points_to_numpy_array(geo)

            controlnet_conditioning_image = torch.from_numpy(
                numpy.array([input_colors])
            ).to(dtype)

            controlnet = diffusers.ControlNetModel.from_pretrained(
                controlnetmodel,
                local_files_only=cache_only,
                torch_dtype=dtype,
            )
            # controlnet.to(torch_device)
            controlnet_models.append(controlnet)
            controlnet_images.append(controlnet_conditioning_image)
            controlnet_scales.append(controlnet_conditioning_scale)

    text_embeddings = torch.from_numpy(numpy.array([numpy.array(input_embeddings["conditional_embedding"]).reshape(
        input_embeddings["tensor_shape"])]))
    
    uncond_embeddings = torch.from_numpy(numpy.array([numpy.array(input_embeddings["unconditional_embedding"]).reshape(
        input_embeddings["tensor_shape"])]))


    input_noise = torch.from_numpy(numpy.array([input_scheduler["noise_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0])])).to(dtype)

    init_latents = torch.from_numpy(numpy.array([input_scheduler["guided_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0])])).to(dtype)
    
    mask_latents = torch.from_numpy(numpy.array([input_scheduler["mask_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0])]))

    from diffusers.utils import load_image
    import PIL

    init_latents = load_image(r"C:\Users\Paul\OneDrive - Bismuth Consultancy B.V\Desktop\TESTFILE.jpg")
    init_latents = [init_latents]
    # resize all images w.r.t passed height an width
    init_latents = [i.resize((512, 512), resample=PIL.Image.LANCZOS) for i in init_latents]
    init_latents = [numpy.array(i.convert("RGB"))[None, :] for i in init_latents]
    init_latents = numpy.concatenate(init_latents, axis=0)

    init_latents = init_latents.transpose(0, 3, 1, 2)
    init_latents = torch.from_numpy(init_latents).to(dtype) / 127.5 - 1.0
    
    mask_latents = load_image(r"C:\Users\Paul\OneDrive - Bismuth Consultancy B.V\Desktop\TESTFILE_MASK.jpg")
    mask_latents = [mask_latents]
    mask_latents = [i.resize((512, 512), resample=PIL.Image.LANCZOS) for i in mask_latents]
    mask_latents = numpy.concatenate([numpy.array(m.convert("L"))[None, None, :] for m in mask_latents], axis=0)
    mask_latents = mask_latents.astype(numpy.float32) / 255.0

    mask_latents[mask_latents < 0.5] = 0
    mask_latents[mask_latents >= 0.5] = 1
    mask_latents = torch.from_numpy(mask_latents)



    model_path = mlops_utils.ensure_huggingface_model_local(model, os.path.join("$MLOPS", "data", "models", "diffusers"), cache_only)
    
    scheduler_config = input_scheduler["config"]
    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(scheduler_config)
    
    pipe = diffusers.StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_path, scheduler=scheduler, controlnet=controlnet_models, torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()

    out_latent = pipe(
        prompt_embeds = text_embeddings,#prompt= "a cat smiling", #
        negative_prompt_embeds = uncond_embeddings,#negative_prompt="bad quality, mangled",
        num_inference_steps=inference_steps,
        guidance_scale = guidance_scale,
        # width=512,
        # height=512,
        eta=1.0,
        #latents = input_noise,
        image=init_latents, # hijacked - CLEAN IMAGE
        mask_image=mask_latents, # hijacked - CLEAN MASK
        control_image=controlnet_images,
        controlnet_conditioning_scale=controlnet_scales,
        output_type = "pil",
        generator = torch.manual_seed(1),
        # TODO: FIX the clamping below. Should not be there, but the solve throws an error otherwise
        strength= max(0.05, image_deviation)
    ).images[0]
    out_latent.show()
    return out_latent
