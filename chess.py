import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import random

imgsize = 40


# rotacija radi ok osim za 33 sliku, pokusati ispraviti to i onda gledati kako da se cropuje slika!
def main():
  # 71 68 54
  for filename in os.listdir('images'):
    print filename
    if filename != "other":
    # if filename == "chess43.png":
      image_color = load_image('images/' + filename)
      display_image(image_color)
      # image_color = erode(image_color, iterations=1)
      # image_color = rotate_chessboard(image_color)
      img = canny(image_color)
      # display_image(img)
      # image1 = find_board(image_color.copy(), img)
      image1 = rotate_chessboard1(image_color)
      # display_image(image1)
      # print(board_box)
      img = invert(image_bin(image_gray(image1)))
      # img = canny(image1)
      selected_regions, numbers = select_outer_figures(image1.copy(), img)
      display_image(selected_regions)
    # for region in regions:
    #   display_image(region)

# def crop_chessboard(image):
#   # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#   # img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   # cnt = contours[0]
#   # x, y, w, h = cv2.boundingRect(cnt)
#   # crop = img[y:y + h, x:x + w]
#   # return crop
#   blurred = cv2.GaussianBlur(image_gray(image), (5, 5), 0)
#   thresh = cv2.threshold(blurred, 0, 127, cv2.THRESH_BINARY)[1]
#   # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#   bin_img = image_bin(image_gray(image))
#   img, contours, hierarchy = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   x, y, w, h = cv2.boundingRect(contours[0])
#   cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#   image = image[y:y + h, x:x + w]
#   return image


def background(image):
  return (image[15, 15] + image[15, image.shape[1] - 15] + image[image.shape[0] - 15, 15] + image[image.shape[0] - 15, image.shape[1] - 15]) / 4


def rotate_chessboard(image):
  lines = hough(image)
  # coskovi
  points = cross_points(lines, image)
  points.sort(key=lambda x: x[0])
  point1 = points[0]
  point2 = points[len(points) - 1]
  points.sort(key=lambda y: y[1])
  point3 = points[0]
  point4 = points[len(points) - 1]
  # cv2.circle(image, (point1[0], point1[1]), 15, (0, 255, 255), -1)
  # cv2.circle(image, (point2[0], point2[1]), 15, (0, 255, 255), -1)
  # cv2.circle(image, (point3[0], point3[1]), 15, (0, 255, 255), -1)
  # cv2.circle(image, (point4[0], point4[1]), 15, (0, 255, 255), -1)

  x = abs(point1[0] - point3[0])
  y = abs(point1[1] - point3[1])
  alpha = -np.arctan2(y, x)
  if y > x:
    alpha = (math.pi / 2 + alpha)
  center = (((point1[0] + point2[0]) / 2), ((point3[1] + point4[1]) / 2))
  M = cv2.getRotationMatrix2D(center, np.degrees(alpha), 1.0)
  image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

  # remove black part
  # image[np.where((image==[0, 0, 0]).all(axis=2))] = [255, 255, 255]
  # image = erode(dilate(image))
  point1 = rotate(point1, alpha, center)
  point2 = rotate(point2, alpha, center)
  point3 = rotate(point3, alpha, center)
  point4 = rotate(point4, alpha, center)
  # cv2.circle(image, (point1[0], point1[1]), 15, (0, 255, 255), -1)
  image = crop_image(point1, point2, point3, point4, image)
  return image


def rotate_chessboard1(image):
  img = canny(image)
  box = find_board(image.copy(), img)
  # point1 = box[0]
  # point2 = box[1]
  # point3 = box[2]
  # point4 = box[3]
  point1 = min(box, key=lambda(b): b[0])
  point2 = max(box, key=lambda(b): b[0])
  point3 = min(box, key=lambda(b): b[1])
  point4 = max(box, key=lambda(b): b[1])
  cv2.circle(image, (point1[0], point1[1]), 5, (255, 0, 255), -1)
  cv2.circle(image, (point2[0], point2[1]), 5, (255, 0, 255), -1)
  cv2.circle(image, (point3[0], point3[1]), 5, (255, 0, 255), -1)
  cv2.circle(image, (point4[0], point4[1]), 5, (255, 0, 255), -1)

  x = abs(point1[0] - point3[0])
  y = abs(point1[1] - point3[1])
  alpha = -np.arctan2(y, x)
  if y > x:
    alpha = (math.pi / 2 + alpha)
  center = (((point1[0] + point2[0]) / 2), ((point3[1] + point4[1]) / 2))
  M = cv2.getRotationMatrix2D(center, np.degrees(alpha), 1.0)
  image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

  point1 = rotate(point1, alpha, center)
  point2 = rotate(point2, alpha, center)
  point3 = rotate(point3, alpha, center)
  point4 = rotate(point4, alpha, center)
  image = crop_image(point1, point2, point3, point4, image)
  return image


def rotate1(point, angle, origin):
    ox = origin[0]
    oy = origin[1]
    px = point[0]
    py = point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def crop_image(point1, point2, point3, point4, image):
  x_min = min(point1[0], point2[0], point3[0], point4[0])
  x_max = max(point1[0], point2[0], point3[0], point4[0])
  y_min = min(point1[1], point2[1], point3[1], point4[1])
  y_max = max(point1[1], point2[1], point3[1], point4[1])
  image = image[y_min:y_max, x_min:x_max]  # obrnute coord -.-'
  return image


def rotate(point, alpha, center):
  cos = np.cos(alpha)
  sin = np.sin(alpha)
  x = (point[1] - center[1]) * cos - (point[0] - center[0]) * sin
  y = (point[1] - center[1]) * sin + (point[0] - center[0]) * cos
  x = x + center[0]
  y = y + center[1]
  return (int(x), int(y))


def cross_points(lines, image):
  points = []
  for line1, line2 in itertools.combinations(lines, 2):
    if line_intersection(line1, line2, image):
      points.append(line_intersection(line1, line2, image))
  # for point in points:
  #   cv2.circle(image, (point[0], point[1]), 5, (255, 0, 255), -1)
  return points


def canny(image):
  gray = image_gray(image)
  edges = cv2.Canny(gray, 100, 300, apertureSize=7)
  return dilate(edges)


def hough(image):
  edges = canny(image)
  hough = cv2.HoughLines(edges, 1, np.pi / 180, 150)
  lines = []

  for h in hough:
    rho = h[0][0]
    theta = h[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1500 * (-b))
    y1 = int(y0 + 1500 * (a))
    x2 = int(x0 - 1500 * (-b))
    y2 = int(y0 - 1500 * (a))
    lines.append(((x1, y1), (x2, y2)))
    cv2.line(image, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 5)
  display_image(image)
  return lines


# preuzeto sa stackoverflow-a
def line_intersection(line1, line2, image):
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

  def det(a, b):
      return a[0] * b[1] - a[1] * b[0]

  div = det(xdiff, ydiff)
  if div == 0:
    return None

  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  if(x > 0 and x < image.shape[1] and y > 0 and y < image.shape[0]):
    return x, y
  else:
    return None


def select_outer_figures(image_orig, image_bin):
  img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  regions_array = []
  hierarchy = hierarchy[0]
  for component in zip(contours, hierarchy):
    contour = component[0]
    hierarchy = component[1]
    if hierarchy[3] == -1:  # izdvoji samo skroz spoljne konture
      x, y, w, h = cv2.boundingRect(contour)
      area = cv2.contourArea(contour)
      if area > 100 and h < 100 and h > 15 and w > 20:
        region = image_bin[y:y + h + 1, x:x + w + 1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
  regions_array = sorted(regions_array, key=lambda item: item[1][0])
  sorted_regions = [r[0] for r in regions_array]
  print "asddff"
  return image_orig, sorted_regions


def find_board(image_orig, image_bin):
  img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, 2)
  for contour in contours:
    area = cv2.contourArea(contour)
    if area < 50000:
      continue
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def resize_region(region):
    return cv2.resize(region, (imgsize, imgsize), interpolation=cv2.INTER_NEAREST)


def load_image(path):
  img = cv2.imread(path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_gray(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
  height, width = image_gs.shape[0:2]
  image_binary = np.ndarray((height, width), dtype=np.uint8)
  ret, image_bin = cv2.threshold(image_gs, 120, 255, cv2.THRESH_BINARY)
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


if __name__ == '__main__':
  main()
