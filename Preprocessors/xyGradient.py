# source: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python/49735873#49735873

import numpy as np
import PIL
import scipy
from scipy import misc, ndimage

def simplePreprocessing(image):
    imageArray = np.array(image)
    imageArray = imageArray.reshape((image.height,image.width,len(image.getbands())))
    img = imageArray.astype('int32')
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    imageFinal = PIL.Image.fromarray(np.uint8(mag))
    return imageFinal
