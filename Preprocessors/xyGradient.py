# source: https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python/49735873#49735873

import numpy as np
import PIL
import scipy
from scipy import misc, ndimage

def xyGradient(img):
    imageData = image.getdata()
    imageArray = np.asarray(imageData)
    imageArray = imageArray.reshape((image.height,image.width,len(image.getbands())))
    img = img.astype('int32')
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1)
    mag =  np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    JpegImageFile(mag)
    return mag

