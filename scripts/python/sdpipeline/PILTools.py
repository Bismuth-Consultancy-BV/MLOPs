import hou
import numpy as np
import math
from PIL import Image

def PointCloudToImage(w, h):
    # Get a reference to the input geometry
    input_geo = hou.pwd().inputGeometry(0)

    # Get the "r", "g", and "b" attributes from the input geometry
    r_attrib = input_geo.pointFloatAttribValues("r")
    g_attrib = input_geo.pointFloatAttribValues("g")
    b_attrib = input_geo.pointFloatAttribValues("b")

    # Remap the range of the "r", "g", and "b" attributes from 0-1 to 0-255
    r_attrib = np.multiply(r_attrib, 255)
    g_attrib = np.multiply(g_attrib, 255)
    b_attrib = np.multiply(b_attrib, 255)

    # Calculate the dimensions of the image based on the length of the "r" attribute
    if(w<0 or h<0):
        num_points = len(r_attrib)
        img_size = int(np.sqrt(num_points))
        img_width = img_size
        img_height = img_size
    else:
        img_width = w
        img_height = h

    # Reshape the color data into a 3D array with the same dimensions as your original image
    cd_array = np.zeros((img_height, img_width, 3))
    cd_array[:,:,0] = np.reshape(r_attrib, (img_height, img_width))
    cd_array[:,:,1] = np.reshape(g_attrib, (img_height, img_width))
    cd_array[:,:,2] = np.reshape(b_attrib, (img_height, img_width))

    # Convert the color data to a PIL image object
    pil_image = Image.fromarray(cd_array.astype('uint8'), 'RGB')

    return pil_image

def ImageToPointCloud(pil_image):
    # Get a reference to the geometry object
    geo = hou.pwd().geometry()

    # Convert the PIL image to a numpy array
    cd_array = np.array(pil_image)

    # Split the color data into separate "r", "g", and "b" arrays
    r_attrib = cd_array[:,:,0].ravel() / 255.0
    g_attrib = cd_array[:,:,1].ravel() / 255.0
    b_attrib = cd_array[:,:,2].ravel() / 255.0

    # Set the "r", "g", and "b" attributes on the points
    geo.setPointFloatAttribValues("r", r_attrib)
    geo.setPointFloatAttribValues("g", g_attrib)
    geo.setPointFloatAttribValues("b", b_attrib)
