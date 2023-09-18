from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
}
