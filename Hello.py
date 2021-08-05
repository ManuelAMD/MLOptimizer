import matplotlib as mat
import matplotlib.pyplot as plt
from app.common.tools import to_monochromatic
from PIL import Image
import numpy as np
import keras.preprocessing.image as k

#img = Image.open('aux/NoMono.png')
img = k.load_img('ResDrought/train20210730-022649_(80, 32, 27)_1d-10h-12m-37s_0.022172659635543823/0.png', target_size=(480,640), color_mode='grayscale')
img = np.array(img)
print(img)
#img *= 255
im = to_monochromatic([img])
print(im)
mat.image.imsave('aux/NoMono2.png', im[0].reshape(480, 640), cmap='Greys')