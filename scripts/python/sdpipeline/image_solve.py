import hou
import mlops_image_utils
import numpy
import torch
import gc
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from tqdm.auto import tqdm
from . import schedulers_lookup


def run(
    inference_steps,
    latent_dimension,
    input_embeddings,
    controlnet_geo,
    attention_slicing,
    guidance_scale,
    input_scheduler,
    torch_device,
    model,
    lora,
    local_cache_only=True,
    seamless_gen=False,
):
    no_half = torch_device in ["mps", "cpu"]
    dtype_unet = torch.float32 if no_half else torch.float16
    dtype_controlnet = numpy.float32 if no_half else numpy.float16
    scheduler_config = input_scheduler["config"]
    t_start = scheduler_config["init_timesteps"]

    scheduler_type = scheduler_config["type"]
    del scheduler_config["init_timesteps"]
    del scheduler_config["type"]
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(
        scheduler_config
    )
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps

    cross_attention_kwargs = {}
    unet = UNet2DConditionModel.from_pretrained(
        model,
        subfolder="unet",
        local_files_only=local_cache_only,
        torch_dtype=dtype_unet,
    )
    if lora["weights"] != "":
        cross_attention_kwargs = {"scale": lora["scale"]}
        unet.load_attn_procs(
            lora["weights"], use_safetensors=lora["weights"].endswith(".safetensors")
        )
    unet.to(torch_device)

    if seamless_gen:
        for module in unet.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = "circular"

    if attention_slicing:
        unet.set_attention_slice("auto")

    mask_latents = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["mask_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    mask_orig = mask_latents.to(torch_device)

    init_latents_orig = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["image_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    init_latents = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["guided_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    noise = torch.from_numpy(
        numpy.array(
            [
                input_scheduler["noise_latent"].reshape(
                    4, latent_dimension[1], latent_dimension[0]
                )
            ]
        )
    ).to(torch_device)
    masking = input_scheduler["image_provided"] == 1

    text_embeddings = numpy.array(input_embeddings["conditional_embedding"]).reshape(
        input_embeddings["tensor_shape"]
    )
    uncond_embeddings = numpy.array(
        input_embeddings["unconditional_embedding"]
    ).reshape(input_embeddings["tensor_shape"])
    text_embeddings = torch.from_numpy(
        numpy.array([uncond_embeddings, text_embeddings])
    ).to(torch_device)

    if not no_half:
        text_embeddings = text_embeddings.half()

    if controlnet_geo:
        controlnet_model = []
        controlnet_image = []
        controlnet_scale = []
        for point in controlnet_geo.points():
            controlnetmodel = point.stringAttribValue("model")
            geo = point.prims()[0].getEmbeddedGeometry()
            controlnet_conditioning_scale = point.attribValue("scale")
            input_colors = mlops_image_utils.colored_points_to_numpy_array(geo)

            controlnet_conditioning_image = torch.from_numpy(
                numpy.array([input_colors])
            ).to(device=torch_device)
            controlnet_conditioning_image = controlnet_conditioning_image.to(dtype_unet)
            controlnet = ControlNetModel.from_pretrained(
                controlnetmodel,
                local_files_only=local_cache_only,
                torch_dtype=dtype_unet,
            )
            controlnet.to(torch_device)
            controlnet_model.append(controlnet)
            controlnet_image.append(controlnet_conditioning_image)
            controlnet_scale.append(controlnet_conditioning_scale)

        controlnet = MultiControlNetModel(controlnet_model)

        if seamless_gen:
            for module in controlnet.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.padding_mode = "circular"

    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    latents = init_latents

    with hou.InterruptableOperation(
        "Solving Stable Diffusion", open_interrupt_dialog=True
    ) as operation:
        # if True:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            latent_model_input = torch.cat([latents] * 2)
            # converting timestep to cpu for compatibility with all samplers
            t = t.to("cpu")
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            ).to(dtype_unet)

            if controlnet_geo:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t.to(dtype_unet),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                )

                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t.to(dtype_unet),
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t.to(dtype_unet),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if masking:
                init_latents_proper = scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * (1.0 - mask_orig)) + (
                    latents * mask_orig
                )

            operation.updateProgress(i / len(timesteps))

    if masking:
        latents = (init_latents_orig * (1.0 - mask_orig)) + (latents * mask_orig)
    latents = latents.cpu().numpy()[0]

    return latents


def run_new(
    inference_steps,
    latent_dimension,
    input_embeddings,
    controlnet_geo,
    attention_slicing,
    cfg_scale,
    scheduler_data,
    torch_device,
    model,
    lora,
    local_cache_only=True,
    seamless_gen=False,
    half=True,
    cleanup = False,
):
    # parameters
    do_image = scheduler_data["image_provided"]
    do_mask = scheduler_data["mask_provided"] and do_image
    do_add_noise = scheduler_data["do_defer"] and do_image    

    do_half = half    
    if torch_device in ["mps", "cpu"]:
        do_half = False
    dtype_torch = torch.float16 if do_half else torch.float32
    dtype_numpy = numpy.float16 if do_half else numpy.float32
    scheduler_config = scheduler_data["config"]

    t_start = 0
    if do_image: # modifying starting timestep when using guide image
        t_start = scheduler_config["init_timesteps"] # start earlier for guidance image
    
    scheduler_type = scheduler_config["type"]    
    scheduler = schedulers_lookup.schedulers[scheduler_type].from_config(
        scheduler_config
    )
    scheduler.set_timesteps(inference_steps)
    timesteps = scheduler.timesteps   
    cross_attention_kwargs = {}
    
    # init models
    unet = UNet2DConditionModel.from_pretrained(
        model,
        subfolder="unet",
        local_files_only=local_cache_only,
        torch_dtype=dtype_torch,
    ).to(torch_device)

    # idk how lora works
    if lora["weights"] != "":
        cross_attention_kwargs = {"scale": lora["scale"]}
        unet.load_attn_procs(
            lora["weights"], use_safetensors=lora["weights"].endswith(
                ".safetensors")
        )

    out = None
    
    # for tiling?
    if seamless_gen:
        for module in unet.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = "circular"

    # optimisation
    if attention_slicing:
        unet.set_attention_slice("auto")

    # setup scheduler
    # text embeddings
    # latent noise / mask / guide
    # denoise

    # text latents
    text_shape = input_embeddings["tensor_shape"]    
    text_embeddings = torch.from_numpy(input_embeddings["conditional_embedding"]).reshape(text_shape)
    uncond_embeddings = torch.from_numpy(input_embeddings["unconditional_embedding"]).reshape(text_shape)        
    text_embeddings = torch.stack([uncond_embeddings, text_embeddings]).to(dtype_torch).to(torch_device)
    # final text_embeddings shape should be [2,77,768]
    

    # image latents
    # 1 x 4 x 64 x 64           
    latent_shape = [1, 4, latent_dimension[1], latent_dimension[0]]    
    noise = torch.from_numpy(scheduler_data["noise_latent"]).reshape(latent_shape).to(torch_device)
    
    if do_image:
        orig_image = torch.from_numpy(scheduler_data["image_latent"]).reshape(latent_shape).to(torch_device)        
        if do_add_noise:
            temp_timesteps = timesteps[t_start].unsqueeze(0)             
            noise = scheduler.add_noise(orig_image, noise, temp_timesteps)       

    # init latents
    latents = noise 
    
    if do_mask:   
        mask_latents = torch.from_numpy(scheduler_data["mask_latent"]).reshape(latent_shape).to(torch_device)   
        #orig_image = image_latents.clone()
        
    # disable this step on certain schedulers otherwise they break only with guideimage on
    # maybe a better way to check for this.. init_noise_sigma value? 
    # sigma is 1.0 on unipc.. maybe we dont need this at all? 14.6 for euler
    if scheduler_config["type"] not in ["LMS", "euler", "euler a"]:  # if unipcmultistep scheduler        
        latents = latents * scheduler.init_noise_sigma    

    timesteps = scheduler.timesteps[t_start:].to(torch_device)    

    if controlnet_geo:
        controlnet_model = []
        controlnet_image = []
        controlnet_scale = []
        for point in controlnet_geo.points():
            controlnetmodel = point.stringAttribValue("model")
            geo = point.prims()[0].getEmbeddedGeometry()
            controlnet_conditioning_scale = point.attribValue("scale")
            input_colors = mlops_image_utils.colored_points_to_numpy_array(geo)

            controlnet_conditioning_image = torch.from_numpy(
                numpy.array([input_colors])
            ).to(device=torch_device)
            controlnet_conditioning_image = controlnet_conditioning_image.to(
                dtype_torch)
            controlnet = ControlNetModel.from_pretrained(
                controlnetmodel,
                local_files_only=local_cache_only,
                torch_dtype=dtype_torch,
            )
            controlnet.to(torch_device)
            controlnet_model.append(controlnet)
            controlnet_image.append(controlnet_conditioning_image)
            controlnet_scale.append(controlnet_conditioning_scale)

        controlnet = MultiControlNetModel(controlnet_model)

        if seamless_gen:
            for module in controlnet.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.padding_mode = "circular"       

    # denoise the latents
    with hou.InterruptableOperation("Solving Stable Diffusion", open_interrupt_dialog=True) as operation:
        # if True:
        for i, t in enumerate(tqdm(timesteps, disable=True)):
            # init new latent model, this has diff shape and just used for predicting unet noise
            latent_model_input = torch.cat([latents] * 2)            
            
            t = t.to("cpu") # converting timestep to cpu for compatibility with all samplers
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)            

            if controlnet_geo:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t.to(dtype_torch),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                )

                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t.to(dtype_torch),
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t,
                                      encoder_hidden_states=text_embeddings,
                                      cross_attention_kwargs=cross_attention_kwargs,).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # increases the impact of noise_pred_text, basically y=mx+c increasing gradient
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample            
            
            if do_mask:                
                # we need to add noise to the original image based on our timestep
                # instead of initializing a new tensor, lets reuse noise_pred 
                noise_pred = scheduler.add_noise(orig_image, noise, t)                             
                latents = (noise_pred * (1.0 - mask_latents)) + (latents * mask_latents)                

            operation.updateProgress(i / len(timesteps))

    if do_mask: # final result only affect masked region
        latents = (orig_image * (1.0 - mask_latents)) + (latents * mask_latents)
    
    out = latents.detach().cpu().squeeze()      

    # cleanup, some articles suggested this but not sure how effective it is
    if torch_device == "cuda" and cleanup:      
        del latents
        del noise
        if do_image:
            del orig_image
        if do_mask:   
            del mask_latents
        gc.collect()
        torch.cuda.empty_cache()

    return out