from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
)

pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    
}
