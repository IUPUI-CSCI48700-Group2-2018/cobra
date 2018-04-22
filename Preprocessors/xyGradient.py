# source: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python/49735873#49735873

import PIL
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage

def xyGradient(img):
    img = img.astype('int32')
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1)
    mag =  np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag
