from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
)

pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
    "StableDiffusionControlNetPipeline": StableDiffusionControlNetPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    "StableDiffusionXLControlNetInpaintPipeline": StableDiffusionXLControlNetInpaintPipeline,
    "StableDiffusionXLControlNetPipeline": StableDiffusionXLControlNetPipeline,
}
