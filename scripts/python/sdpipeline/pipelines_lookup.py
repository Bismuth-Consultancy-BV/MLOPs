from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,

    StableDiffusionXLControlNetPipeline,
)

pipelines = {
    "StableDiffusionPipeline": StableDiffusionPipeline,
    "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
    "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
    "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
    "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    "StableDiffusionXLControlNetInpaintPipeline": StableDiffusionXLControlNetInpaintPipeline,

    "StableDiffusionXLControlNetPipeline": StableDiffusionXLControlNetPipeline,
}
