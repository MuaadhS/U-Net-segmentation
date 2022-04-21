# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 07:35:19 2022

@author: moaat
"""

import time

start = time.time()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.inference_engine import IECore
from patchify import patchify, unpatchify

####
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import logging

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)
log = logging.getLogger()

from arm import *
init() # initialize robotic arm position

####
from pushbullet import PushBullet

API_KEY = "o.ZjGBEbhflDmjeLjFlz1lhTwNFSJUIFSN"

def notify():
    pb = PushBullet(API_KEY)

    text = conf

    push = pb.push_note('The sample is ready!', text)

    with open("mask.png", "rb") as pic:
        file_data = pb.upload_file(pic, "picture.png")

    push = pb.push_file(**file_data)


def build_argparser():    
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                          required=True, type=Path)

    
    args.add_argument('-i', '--input', required=True,
                          help='Required. An input to process. The input must an image.')

    args.add_argument('-d', '--device', default='CPU', type=str,
                          help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                               'acceptable. The demo will look for a suitable plugin for device specified. '
                               'Default value is CPU.')
    
    return parser

####

def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes=False
) -> np.ndarray:
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
    :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
    :param remove_holes: If True, remove holes in the segmentation result.
    :return: An RGB image where each pixel is an int8 value according to colormap.
    """
    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    if len(np.unique(result)) > colormap.shape[0]:
        raise ValueError(
            f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask

def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.
    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).
    # Returns
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

####
args = build_argparser().parse_args()
####

# Initialize inference engine core and read the IR model 
log.info('Initializing Inference Engine...')
ie = IECore()

net = ie.read_network(
    model=args.model)
    #model="model/saved_model.xml")
    
exec_net = ie.load_network(net, args.device)
#exec_net = ie.load_network(net, "CPU")

output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))




# Configure the input 
image = cv2.imread(args.input, 0)
#image = cv2.imread(r"d1-c1-002.jpg", 0)
N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims

image_norm = np.expand_dims(normalize(np.array(image), axis=1),2)
image_norm = image_norm[:,:,0][:,:,None]
input_image = np.expand_dims(image_norm.transpose(2, 0, 1), 0)

patches = patchify(image, (512, 512), step=512)

image_h, image_w, = image.shape

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):

        # Run the infernece
        single_patch = patches[i,j,:,:]
        image_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        image_norm = image_norm[:,:,0][:,:,None]
        input_image = np.expand_dims(image_norm.transpose(2, 0, 1), 0)
        result = exec_net.infer(inputs={input_layer_ir: input_image})
        result_ir = result[output_layer_ir]
         
        
        #Predict and threshold for values above 0.5 probability
        single_patch_prediction = (result_ir[0,0,:,:] > 0.5).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)


predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 512,512) )

segmentation_mask = unpatchify(predicted_patches_reshaped, image.shape)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




# Define colormap, each color represents a class
colormap = np.array([[68, 1, 84], [255, 216, 52]])


# Define the transparency of the segmentation mask on the photo
alpha = 0.3

# Transform mask to an RGB image
mask = segmentation_map_to_image(segmentation_mask, colormap)
resized_mask = cv2.resize(mask, (image_w, image_h))

# Create image with mask put on
image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)

#######

cv2.imwrite('mask.png', image_with_mask)

numberPixels = len(cv2.findNonZero(segmentation_mask))
#print(numberPixels)

img_area = segmentation_mask.shape[0] * segmentation_mask.shape[1]
segmented_cells = (1-((img_area - numberPixels)/ img_area)) * 100


conf_text=("The estimated confluency is (%): ")

conf = conf_text + str(segmented_cells)

print(conf)

if segmented_cells >= 80:
    notify()
    motion()

#######
'''
#Show segmentation output 
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.title('Image')
plt.imshow(image, cmap='gray')
plt.subplot(232)
plt.title('Mask')
plt.imshow(segmentation_mask, cmap='viridis')
plt.subplot(233)
plt.title('Image with mask')
plt.imshow(image_with_mask)
'''
end = time.time()
print("The elapsed time is (s): " + str(end - start))

#plt.show()
