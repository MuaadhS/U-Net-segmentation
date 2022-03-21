# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:19:02 2022

@author: moaat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:37:14 2022

@author: moaat
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.inference_engine import IECore



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


# Initialize inference engine core and read the IR model 
ie = IECore()

net = ie.read_network(
    model="model/saved_model.xml")
    
exec_net = ie.load_network(net, "CPU")

output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))




# Configure the input 
image = cv2.imread(r"image__3_00.tif", 0)
#image = cv2.imread(r"d1-c1-002.jpg", 0)


N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims

#resized_image = cv2.resize(image, (512, 512))
image = cv2.resize(image, (512, 512))
image_h, image_w, = image.shape

image_norm = np.expand_dims(normalize(np.array(image), axis=1),2)
image_norm = image_norm[:,:,0][:,:,None]
input_image = np.expand_dims(image_norm.transpose(2, 0, 1), 0)


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image, cmap = 'gray')



# Run the infernece
result = exec_net.infer(inputs={input_layer_ir: input_image})
result_ir = result[output_layer_ir]
 
# Prepare data for visualization
segmentation_mask=(result_ir[0,0,:,:] > 0.5).astype(np.uint8)


plt.imshow(segmentation_mask, cmap='viridis')



# Define colormap, each color represents a class
colormap = np.array([[68, 1, 84], [255, 216, 52]])#, [53, 183, 120], [199, 216, 52]])
#colormap = np.array([[0, 0, 0], [255, 255, 255]])  68, 1, 84 purple ,  48, 103, 141 skyblue , 53, 183, 120 green, 199, 216, 52 yellow


# Define the transparency of the segmentation mask on the photo
alpha = 0.3

# Transform mask to an RGB image
mask = segmentation_map_to_image(segmentation_mask, colormap)
resized_mask = cv2.resize(mask, (image_w, image_h))

# Create image with mask put on
image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)


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

