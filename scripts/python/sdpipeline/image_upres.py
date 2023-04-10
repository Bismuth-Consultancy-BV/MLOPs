from diffusers import StableDiffusionUpscalePipeline
import torch
import numpy

# ## Helper functions
# def load_image(p):
#     '''
#     Function to load images from a defined path
#     '''
#     from PIL import Image
#     return Image.open(p).convert('RGB').resize((128,128))

# def numpy_to_pil(images):
#     """
#     Convert a numpy image or a batch of images to a PIL image.
#     """
#     from PIL import Image

#     if images.ndim == 3:
#         images = images[None, ...]
#     images = (images * 255).round().astype("uint8")
#     if images.shape[-1] == 1:
#         # special case for grayscale (single channel) images
#         pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
#     else:
#         pil_images = [Image.fromarray(image) for image in images]

#     return pil_images

def run(positive_embeds, negative_embeds, input_image, steps, seed, latent_dimension, model, torch_device, local_cache_only):
    generator = torch.manual_seed(seed)
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model, local_files_only=local_cache_only)
    pipeline.enable_attention_slicing("max")
    pipeline = pipeline.to(torch_device)

    positive_embeds = torch.from_numpy(numpy.array(positive_embeds).reshape(1, latent_dimension[0], latent_dimension[1])).to(torch_device)
    negative_embeds = torch.from_numpy(numpy.array(negative_embeds).reshape(1, latent_dimension[0], latent_dimension[1])).to(torch_device)

    #input_image = load_image(r"C:\Users\Paul\Downloads\mo.jpg") # remove 
    #input_image = numpy.array(input_image).astype(numpy.float32) / 255.0 # remove

    input_image = numpy.array([input_image])
    #print(input_image.shape)
    #input_image = input_image.transpose(0, 3, 1, 2) # maybe
    #input_image = (input_image * 2.0) - 1.0
    input_image = torch.from_numpy(input_image)

    upscaled_image = pipeline(prompt="", prompt_embeds=positive_embeds, num_inference_steps=steps, output_type="np", negative_prompt_embeds=negative_embeds, image=input_image).images[0]    
    
    #image = numpy_to_pil(upscaled_image)[0].show() # remove

    return upscaled_image