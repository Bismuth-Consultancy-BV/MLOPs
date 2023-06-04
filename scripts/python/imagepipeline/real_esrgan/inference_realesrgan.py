import os

import hou
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from imagepipeline.real_esrgan.realesrgan.archs.srvgg_arch import SRVGGNetCompact
from imagepipeline.real_esrgan.realesrgan.utils import RealESRGANer


def run(
    input_colors,
    model_name,
    model_path,
    denoise_strength,
    outscale,
    tile,
    tile_pad,
    pre_pad,
    netscale,
    face_enhance,
    fp32,
):
    gpu_id = None

    model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
    # determine models according to model names
    model_name = model_name.split(".")[0]
    if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]

    if model_name == "custom":
        if not os.path.isfile(model_path):
            raise hou.NodeError("The specified model path is not valid!")
    else:
        model_path = os.path.join(
            hou.text.expandString("$MLOPS_MODELS"), "real_esrgan", model_name + ".pth"
        )
        if not os.path.isfile(model_path):
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(
                        hou.text.expandString("$MLOPS_MODELS"), "real_esrgan"
                    ),
                    progress=True,
                    file_name=None,
                )

    # use dni to control the denoise strength
    dni_weight = None
    if model_name == "realesr-general-x4v3" and denoise_strength != 1:
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id,
    )

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer

        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    input_colors = input_colors.transpose(1, 2, 0)

    if face_enhance:
        _, _, output = face_enhancer.enhance(
            input_colors, has_aligned=False, only_center_face=False, paste_back=True
        )
    else:
        output, _ = upsampler.enhance(input_colors, outscale=outscale)

    return output
