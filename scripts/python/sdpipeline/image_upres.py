from diffusers import StableDiffusionUpscalePipeline
import torch
import numpy


## Helper functions
def load_image(p):
    '''
    Function to load images from a defined path
    '''
    from PIL import Image
    return Image.open(p).convert('RGB').resize((128,128))


def run(positive_embeds, negative_embeds, input_image, latent_dimension, model, torch_device, local_cache_only):

    # load model and scheduler
    model_id = ""
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model, local_files_only=local_cache_only)
    pipeline.enable_attention_slicing("max")
    pipeline = pipeline.to(torch_device)

    #print(latent_dimension)

    positive_embeds = torch.from_numpy(numpy.array(positive_embeds).reshape(1, latent_dimension[0], latent_dimension[1])).to(torch_device)
    negative_embeds = torch.from_numpy(numpy.array(negative_embeds).reshape(1, latent_dimension[0], latent_dimension[1])).to(torch_device)
    #print(positive_embeds.shape)
    #text_embeddings = torch.from_numpy(numpy.array([uncond_embeddings, text_embeddings])).to(torch_device)

    #input_image = torch.from_numpy(numpy.array([input_image]))
    #print(input_image.shape)

    #upscaled_image = pipeline(prompt="", prompt_embeds=positive_embeds, negative_prompt_embeds=negative_embeds, image=input_image).images[0]
    upscaled_image = pipeline(prompt="man", num_inference_steps = 20, output_type="np", image=load_image(r"C:\Users\Paul\Downloads\mo.jpg")).images[0]
    
    return upscaled_image