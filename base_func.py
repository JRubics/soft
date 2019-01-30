import cv2
import matplotlib.pyplot as plt
import numpy as np


imgsize = 40


def resize_region(region):
    return cv2.resize(region, (imgsize, imgsize), interpolation=cv2.INTER_NEAREST)


def load_image(path):
  img = cv2.imread(path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_gray(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
  height, width = image_gs.shape[0:2]
  ret, image_bin = cv2.threshold(image_gs, 110, 255, cv2.THRESH_BINARY)
  return image_bin


def invert(image):
  return 255 - image


def display_image(image, color=False):
  if color:
    plt.imshow(image)
  else:
    plt.imshow(image, 'gray')
  plt.show()


def dilate(image, iterations=1):
  kernel = np.ones((3, 3))
  return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, iterations=1):
  kernel = np.ones((3, 3))
  return cv2.erode(image, kernel, iterations=iterations)


def canny(image, threshold1=100, threshold2=300, apertureSize=3):
  gray = image_gray(image)
  edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)
  return dilate(edges)
