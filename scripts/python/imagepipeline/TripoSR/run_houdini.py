import argparse
import os
import tempfile
import uuid

import numpy as np
import rembg
import torch
from PIL import Image

import sys
sys.path.insert(0,os.path.join("$HIP",'../../scripts/python/imagepipeline/TripoSR'))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

import mlops_image_utils
import mlops_utils

from importlib import reload
reload(mlops_image_utils)
reload(mlops_utils)


# Inputs
node = hou.pwd()
geo = node.geometry()
geo0 = node.inputs()[0].geometry()
geo3 = node.inputs()[0].geometry()

# Parameters
device = node.parent().parm("device").evalAsString()
model = "stabilityai/TripoSR"
parser = argparse.ArgumentParser()


chunk_size=4096         # "Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
mc_resolution=256       # "Marching cubes grid resolution. Default: 256"
no_remove_bg=False      # "If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
foreground_ratio=0.85   #"Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",

if not torch.cuda.is_available():
    device = "cpu"

torch.cuda.empty_cache()

model = TSR.from_pretrained(
    model,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)


image = mlops_image_utils.colored_points_to_numpy_array(geo0)
image = mlops_image_utils.colors_numpy_array_to_pil(image)

if not no_remove_bg:
    rembg_session = rembg.new_session()

    image = remove_background(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))



with torch.no_grad():
    scene_codes = model([image], device=device)

output_dir=os.path.join(tempfile.gettempdir(),'mlops_image_to_3d')
os.makedirs(output_dir, exist_ok=True)
mesh_file=os.path.join(output_dir, f"mesh_{uuid.uuid4()}.obj")

meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)
meshes[0].export(mesh_file)

geo.setGlobalAttribValue("output_dir",output_dir)
geo.setGlobalAttribValue("mesh_file",mesh_file)

try:
    _obj = hou.node("../obj_importer1")
   #_obj.cook()
    #_source = _obj.parm("sObjFile")
    #_source.set(mesh_file)
    
except Exception as e:
    print(e)