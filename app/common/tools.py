import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import pandas as pd
from pandas import DataFrame
import matplotlib as mat
import tempfile
import shutil
import warnings
import PIL
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
import cv2
import sys

def get_names(file: str):
    names = DataFrame(pd.read_csv(file, header=None))
    return names

def load_imgs(data_folder: str, names_file: str, rows: int, cols: int, channels=1, img_type = '.png', color_mode='grayscale'):
    names = get_names(names_file)
    x = []
    for name in names[0]:
        img = image.load_img(data_folder+'/'+str(name)+img_type, color_mode=color_mode, target_size=(rows, cols, channels))
        img = np.array(img)
        x.append(img)
    x = np.array(x)
    return x

def to_monochromatic(img_data):
    x_mono = []
    for i in img_data:
        (thresh, monoImg) = cv2.threshold(i, 130, 140, cv2.THRESH_BINARY)
        x_mono.append(monoImg)
    x_mono = np.array(x_mono)
    return x_mono

def image_evaluation(img_org, img_new, width, height):
    error = 0
    accuracy = 0
    for i in range(width):
        for j in range(height):
            if img_org[i,j] != img_new[i,j]:
                error += 1
            else:
                accuracy += 1
    tam = width * height
    err = error/tam
    acc = accuracy/tam
    return err, acc

#Toma todos los colores existentes en la imagen
def get_colors(image):
  aux = []
  band = True
  for i in image:
    for j in i:
      for k in aux:
        if j.tolist() == k:
          band = False
          break
      if band:
        aux.append(j.tolist())
      band = True
  return np.array(aux)

#Obtiene los n colores principales mediante kmeans
def principal_colors(img, n_colors):
  photo = img
  photo = photo/255
  w,h,d = tuple(photo.shape)
  assert d == 3
  photo = photo.reshape(w*h, d)
  photo_sample = shuffle(photo, random_state=0)[:1000]
  kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(photo_sample)
  labels = kmeans.predict(photo)
  return labels, kmeans

#Recrea una imagen comprimida por la función de kmaens por un codebook
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

#Función para transformar y obtener los colores principales de una imagen
def n_colors_img(img, n_colors):
  labels, model = principal_colors(img, n_colors)
  w,h,d = tuple(img.shape)
  res = recreate_image(model.cluster_centers_, labels, w, h)
  res = np.round(res*255).astype(np.uint8)
  return res

#Función para dada una paleta solo tomar los colores de esa paleta en la imagen
def quantizetopalette(silf, palette, dither=False):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  palette.load()
  im = silf.im.convert("P", 0, palette.im)
  # the 0 above means turn OFF dithering making solid colors
  return silf._new(im)

#Realiza las operaciones necesarias para obtener una imagen RGB por una paleta de colores
def rgb_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('P', (rows, cols))
  palImage.putpalette(palettedata*32)
  #palImage.putpalette(palettedata)
  oldImage = Image.fromarray((img*255).astype(np.uint8)).convert("RGB")
  newImage = quantizetopalette(oldImage,palImage)
  res_image = np.asarray(newImage.convert("RGB"))
  return res_image

#Transforma una imagen en escala de grises a RGB dada una paleta de colores
def gray_to_rgb_by_pallete(img, palette):
  img_float32 = np.float32(palette)
  palette_gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
  rows = len(img)
  cols = len(img[0])
  new_img = np.zeros((rows, cols,3)) #3 color channels
  for i in range(rows):
    for j in range(cols):
      index = np.argmin(np.abs(palette_gray-img[i][j]))
      new_img[i][j] = palette[index]
  return new_img

def rgb_to_gray(data):
  gray_data = []
  for i in data:
    gray_data.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
  gray_data = np.array(gray_data)
  return gray_data

#Realiza las operaciones necesarias para obtener una imagen RGB por una paleta de colores
def gray_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  #print(palettedata)
  palImage = Image.new('P', (rows, cols))
  #print(palImage)
  palImage.putpalette(palettedata*32)
  #palImage.putpalette(palettedata)
  #print(img)
  oldImage = Image.fromarray((img*255).astype(np.uint8))#.convert("P")
  newImage = quantizetopalette(oldImage,palImage)
  res_image = np.asarray(newImage)#.conv

#Toma todos los colores existentes en la imagen
def get_colors(image):
  aux = []
  band = True
  for i in image:
    for j in i:
      for k in aux:
        if j.tolist() == k:
          band = False
          break
      if band:
        aux.append(j.tolist())
      band = True
  return np.array(aux)

#Obtiene los n colores principales mediante kmeans
def principal_colors(img, n_colors):
  photo = img
  photo = photo/255
  w,h,d = tuple(photo.shape)
  assert d == 3
  photo = photo.reshape(w*h, d)
  photo_sample = shuffle(photo, random_state=0)[:1000]
  kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(photo_sample)
  labels = kmeans.predict(photo)
  return labels, kmeans

#Recrea una imagen comprimida por la función de kmaens por un codebook
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

#Función para transformar y obtener los colores principales de una imagen
def n_colors_img(img, n_colors):
  labels, model = principal_colors(img, n_colors)
  w,h,d = tuple(img.shape)
  res = recreate_image(model.cluster_centers_, labels, w, h)
  res = np.round(res*255).astype(np.uint8)
  return res

#Función para dada una paleta solo tomar los colores de esa paleta en la imagen
def quantizetopalette(silf, palette, dither=False):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  palette.load()
  im = silf.im.convert("P", 0, palette.im)
  # the 0 above means turn OFF dithering making solid colors
  return silf._new(im)

#Realiza las operaciones necesarias para obtener una imagen RGB por una paleta de colores
def rgb_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('P', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img).convert("RGB")
  newImage = quantizetopalette(oldImage,palImage)
  res_image = np.asarray(newImage.convert("RGB"))
  return res_image

#Transforma una imagen en escala de grises a RGB dada una paleta de colores
def gray_to_rgb_by_pallete(img, palette):
  img_float32 = np.float32(palette)
  palette_gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
  rows = len(img)
  cols = len(img[0])
  new_img = np.zeros((rows, cols,3)) #3 color channels
  for i in range(rows):
    for j in range(cols):
      index = np.argmin(np.abs(palette_gray-img[i][j]))
      new_img[i][j] = palette[index]
  return new_img

def divisors(num: int):
    res = []
    for i in range(1, int(num/2)+1):
        if num % i == 0:
            res.append(i)
    return res

def divisors_tuple(num: int):
    res = []
    for i in range(1, int(num/2)+1):
        if num % i == 0:
            res.append(i)
    return tuple(res)
