# DONE
######## TEXT PROMPT NODE ########
ATTR_PROMPT = ["dog"]
ATTR_MODEL = r"C:\BISMUTH\Work\Projects\Monaco\checkpoints\SD_2_1\stable-diffusion-2-1"

######## TEXT PROMPT NODE ########

# DONE
###### TOKENIZER NODE ######
from sdpipeline import tokenizer
ATTR_INPUT_IDS = tokenizer.run(ATTR_PROMPT, model=ATTR_MODEL, local_cache_only=True)
ATTR_INPUT_IDS_NEG = tokenizer.run("", model=ATTR_MODEL, local_cache_only=True)

# DONE
###### EMBEDDER NODE  ######
from sdpipeline import embedder
ATTR_TORCH_DEVICE = "cuda"
# run(input_ids_positive, input_ids_negative, torch_device, model="openai/clip-vit-large-patch14", local_cache_only=True)
ATTR_EMBEDDINGS = embedder.run(ATTR_INPUT_IDS, ATTR_INPUT_IDS_NEG, ATTR_TORCH_DEVICE, model=ATTR_MODEL, local_cache_only=True)


from sdpipeline import image_upres

negative_embeds = ATTR_EMBEDDINGS["unconditional_embedding"]
positive_embeds = ATTR_EMBEDDINGS["conditional_embedding"]
latent_dimension = ATTR_EMBEDDINGS["tensor_shape"]
input_image = None
model = "stabilityai/stable-diffusion-x4-upscaler"
torch_device = "cuda"
local_cache_only = True
seed = 44
steps = 20

#ATTR_UPRES = image_upres.run(positive_embeds, negative_embeds, input_image, steps, seed, latent_dimension, model, torch_device, local_cache_only)



# DONE
###### LATENT NOISE NODE  ######
from sdpipeline import latent_noise
ATTR_SEED = 44                                 # Seed
ATTR_IMAGE_HEIGHT = 512                        # default ATTR_IMAGE_HEIGHT of Stable Diffusion
ATTR_IMAGE_WIDTH = 512                         # default ATTR_IMAGE_WIDTH of Stable Diffusion
ATTR_TORCH_DEVICE = "cuda"
ATTR_LATENTS = latent_noise.run(ATTR_SEED, ATTR_IMAGE_WIDTH, ATTR_IMAGE_HEIGHT, ATTR_TORCH_DEVICE)

# # DONE
# ###### SCHEDULER NODE #######
# from sdpipeline import scheduler
# ATTR_NUM_INFERENCE_STEPS = 20                  # Number of denoising steps
# ATTR_TORCH_DEVICE = "cuda"
# ATTR_SCHEDULER_MODEL = "unipc"
# ATTR_SCHEDULER_DATA = scheduler.run(ATTR_LATENTS, ATTR_TORCH_DEVICE)
# ATTR_GUIDING_STRENGTH = 0.8
# run(input_latents, latent_dimension, guided_latents, ATTR_GUIDING_STRENGTH, 0.0, ATTR_NUM_INFERENCE_STEPS, ATTR_TORCH_DEVICE, ATTR_SCHEDULER_MODEL, model=ATTR_MODEL, local_cache_only=True)

# # DONE
# ##### UNET SOLVER NODE ########
# from sdpipeline import image_solve
# ATTR_NUM_INFERENCE_STEPS = 20                  # Number of denoising steps
# ATTR_TORCH_DEVICE = "cuda"
# ATTR_SOLVED_LATENTS = image_solve.run(ATTR_NUM_INFERENCE_STEPS, ATTR_EMBEDDINGS, ATTR_SCHEDULER_DATA, ATTR_TORCH_DEVICE, model=ATTR_MODEL, local_cache_only=True)

# # DONE
# ######## IMAGE DECODER NODE ########
# from sdpipeline import image_decode
# ATTR_TORCH_DEVICE = "cuda"
# ATTR_IMAGE = image_decode.run(ATTR_SOLVED_LATENTS, ATTR_TORCH_DEVICE, model=ATTR_MODEL, local_cache_only=True)

# # ##### UTILITY TO WRITE IMAGE TO DISK #######
# # from sdpipeline import image_export
# # path = r"C:\Users\Paul\OneDrive - Bismuth Consultancy B.V\BISMUTH_CONSULTANCY\Work\Projects\Monaco\Toolset\MLOPs\scripts\pythongeeks.jpg"
# # image_export.run(ATTR_IMAGE, ATTR_IMAGE_WIDTH, ATTR_IMAGE_HEIGHT, path)
