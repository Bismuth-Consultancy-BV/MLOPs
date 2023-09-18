from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
}
