# DONE
######## TEXT PROMPT NODE ########
ATTR_PROMPT = ["a photograph of an astronaut riding a horse"]
######## TEXT PROMPT NODE ########

# DONE
###### TOKENIZER NODE ######
from sdpipeline import tokenizer
ATTR_INPUT_IDS = tokenizer.run(ATTR_PROMPT, model="openai/clip-vit-large-patch14", local_cache_only=True)


###### EMBEDDER NODE  ######
from sdpipeline import embedder
ATTR_TORCH_DEVICE = "cuda"
ATTR_EMBEDDINGS = embedder.run(ATTR_INPUT_IDS, ATTR_TORCH_DEVICE, model="openai/clip-vit-large-patch14", local_cache_only=True)

###### LATENT NOISE NODE  ######
from sdpipeline import latent_noise
ATTR_SEED = 44                                 # Seed
ATTR_IMAGE_HEIGHT = 512                        # default ATTR_IMAGE_HEIGHT of Stable Diffusion
ATTR_IMAGE_WIDTH = 512                         # default ATTR_IMAGE_WIDTH of Stable Diffusion
ATTR_TORCH_DEVICE = "cuda"
ATTR_LATENTS = latent_noise.run(ATTR_PROMPT, ATTR_SEED, ATTR_IMAGE_WIDTH, ATTR_IMAGE_HEIGHT, ATTR_TORCH_DEVICE)

###### SCHEDULER NODE #######
from sdpipeline import scheduler
ATTR_TORCH_DEVICE = "cuda"
ATTR_SCHEDULER_DATA = scheduler.run(ATTR_LATENTS, ATTR_TORCH_DEVICE)

##### UNET SOLVER NODE ########
from sdpipeline import image_solve
ATTR_NUM_INFERENCE_STEPS = 20                  # Number of denoising steps
ATTR_TORCH_DEVICE = "cuda"
ATTR_SOLVED_LATENTS = image_solve.run(ATTR_NUM_INFERENCE_STEPS, ATTR_EMBEDDINGS, ATTR_SCHEDULER_DATA, ATTR_TORCH_DEVICE, model="CompVis/stable-diffusion-v1-4", local_cache_only=True)

######## IMAGE DECODER NODE ########
from sdpipeline import image_decode
ATTR_TORCH_DEVICE = "cuda"
ATTR_IMAGE = image_decode.run(ATTR_SOLVED_LATENTS, ATTR_TORCH_DEVICE, model="CompVis/stable-diffusion-v1-4", local_cache_only=True)

##### UTILITY TO WRITE IMAGE TO DISK #######
from sdpipeline import image_export
path = r"C:\Users\Paul\OneDrive - Bismuth Consultancy B.V\BISMUTH_CONSULTANCY\Work\Projects\Monaco\Toolset\MLOPs\scripts\pythongeeks.jpg"
image_export.run(ATTR_IMAGE, ATTR_IMAGE_WIDTH, ATTR_IMAGE_HEIGHT, path)
