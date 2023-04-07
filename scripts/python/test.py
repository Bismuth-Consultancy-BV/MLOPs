# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# import torch

# model_id = "stabilityai/stable-diffusion-2-1"


# from transformers import CLIPTokenizer
# CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=False)


# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
    
# image.save("astronaut_rides_horse.png")

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="stabilityai/stable-diffusion-2-1", repo_type="model")
