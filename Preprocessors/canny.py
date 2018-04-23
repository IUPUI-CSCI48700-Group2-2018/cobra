import numpy as np
from PIL import Image
import skimage
from skimage import feature
import matplotlib as plt

# load color image
img_rgb = '/Users/Amos/Desktop/FH.png'
img_arr = np.array(Image.open(img_rgb), dtype=np.uint8)

img_arr.shape
(1005, 740, 3)

# convert to grayscale image
from skimage.color import rgb2gray
img_gray = rgb2gray(img_arr)

img_gray.shape
(1005, 740)

edges1 = feature.canny(img_gray)
edges2 = feature.canny(img_gray, sigma=3)

edges1.shape
(1005, 740)

edges2.shape
(1005, 740)

# display    
plt.imshow(edges1)