import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import random
import sys
# keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import model_from_json

imgsize = 40
photo = 0
find_alpha = 0


def main(train):
  for filename in os.listdir('images'):
    # if filename != "other" and filename != "phone" and filename != "computer" and filename != "train":
    if filename == "chess65.png":
      print filename
      if photo is 1:
        image_color = load_image('images/' + filename)
        image_color = dilate(erode(image_color, 2), 2)
        img = image_bin(image_gray(image_color))
        image_color = find_chessboard(image_color, img)
        selected_regions, regions = find_figures(image_color.copy())
        # print(len(regions))
        # display_image(selected_regions)
      else:
        image_color = load_image('images/' + filename)
        image_color = dilate(erode(image_color, iterations=1))
        image_color = find_chessboard(image_color, image_color)
        # display_image(image_color)
        selected_regions, regions = find_figures(image_color.copy())

        network(train, regions)

        display_image(selected_regions)
        # for region in regions:
        #   display_image(region)


# R = 0, H = 1, B = 2, Q = 3, K = 4, P = 5
def create_alphabet():
  global find_alpha
  find_alpha = 1
  figures = []
  alphabet = []

  image_color = load_image('images/train/chess1.png')
  image_color = dilate(erode(image_color, iterations=1))
  selected_regions, regions = find_model_figures(image_color.copy())
  figures += regions
  alphabet += [0, 1, 2, 3, 4, 2, 1, 0]

  image_color = load_image('images/train/chess2.png')
  image_color = dilate(erode(image_color, iterations=1))
  selected_regions, regions = find_model_figures(image_color.copy())
  figures += regions
  alphabet += [5, 5]

  image_color = load_image('images/train/chess3.png')
  image_color = dilate(erode(image_color, iterations=1))
  selected_regions, regions = find_model_figures(image_color.copy())
  figures += regions
  alphabet += [5, 5]

  image_color = load_image('images/train/chess4.png')
  image_color = dilate(erode(image_color, iterations=1))
  selected_regions, regions = find_model_figures(image_color.copy())
  figures += regions
  alphabet += [0, 1, 2, 3, 4, 2, 1, 0]

  find_alpha = 0
  return figures, alphabet


def network(train, regions):
  figures, alphabet = create_alphabet()
  if train == "train":
    inputs = prepare_for_ann(figures)
    outputs = convert_output(alphabet)
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)

    save_model(ann)
  if train == "no":
    ann = read_model()

  input_regions = prepare_for_ann(regions)
  result = ann.predict(np.array(input_regions[0:], np.float32))
  # print(result)

  result_vector = display_result(result, alphabet)
  # for idx, res in enumerate(result_vector):
  #   print(res)
  #   display_image(regions[idx])
  print(result_vector)


def read_model():
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  ann = model_from_json(loaded_model_json)
  ann.load_weights("model.h5")
  print("Loaded model from disk")
  return ann


def save_model(ann):
  model_json = ann.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  ann.save_weights("model.h5")
  print("Saved model to disk")


def scale_to_range(image):
  return image / 255


def matrix_to_vector(image):
  return image.flatten()


def prepare_for_ann(regions):
  ready_for_ann = []
  for region in regions:
    scale = scale_to_range(region)
    ready_for_ann.append(matrix_to_vector(scale))

  return ready_for_ann


def convert_output(alphabet):
  nn_outputs = []
  for index in range(len(alphabet)):
      output = np.zeros(len(alphabet))
      output[index] = 1
      nn_outputs.append(output)
  return np.array(nn_outputs)


def create_ann():
  ann = Sequential()
  ann.add(Dense(128, input_dim=1600, activation='sigmoid'))
  ann.add(Dense(20, activation='sigmoid'))
  return ann


def train_ann(ann, input_train, output_train):
    input_train = np.array(input_train, np.float32)
    output_train = np.array(output_train, np.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(input_train, output_train, epochs=150, batch_size=1, shuffle=False) 
    return ann


def winner(output):
  return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
  result = []
  for output in outputs:
      result.append(alphabet[winner(output)])
  return result


# def rotate_chessboard(orig, image):
#   lines = hough(image)
#   # coskovi
#   points = cross_points(lines, image)
#   points.sort(key=lambda x: x[0])
#   point1 = points[0]
#   point2 = points[len(points) - 1]
#   points.sort(key=lambda y: y[1])
#   point3 = points[0]
#   point4 = points[len(points) - 1]
#   # cv2.circle(image, (point1[0], point1[1]), 15, (0, 255, 255), -1)

#   x = abs(point1[0] - point3[0])
#   y = abs(point1[1] - point3[1])
#   alpha = -np.arctan2(y, x)
#   if y > x:
#     alpha = (math.pi / 2 + alpha)
#   center = (((point1[0] + point2[0]) / 2), ((point3[1] + point4[1]) / 2))
#   M = cv2.getRotationMatrix2D(center, np.degrees(alpha), 1.0)
#   orig = cv2.warpAffine(orig, M, (image.shape[1], image.shape[0]))

#   point1 = rotate(point1, alpha, center)
#   point2 = rotate(point2, alpha, center)
#   point3 = rotate(point3, alpha, center)
#   point4 = rotate(point4, alpha, center)
#   orig = crop_image(point1, point2, point3, point4, orig)
#   return orig


def find_chessboard(orig, image):
  if photo is 1:
    img = canny_gray(image)
  else:
    img = canny(image)

  box = find_board_box(image.copy(), img)

  point1 = min(box, key=lambda(b): b[0])
  point2 = max(box, key=lambda(b): b[0])
  point3 = min(box, key=lambda(b): b[1])
  point4 = max(box, key=lambda(b): b[1])
  # cv2.circle(image, (point1[0], point1[1]), 5, (255, 0, 255), -1)

  orig, alpha, center = rotate_image(point1, point2, point3, point4, orig)

  point1 = rotate(point1, alpha, center)
  point2 = rotate(point2, alpha, center)
  point3 = rotate(point3, alpha, center)
  point4 = rotate(point4, alpha, center)

  orig = crop_image(point1, point2, point3, point4, orig)

  return orig


def rotate_image(point1, point2, point3, point4, image):
  x = abs(point1[0] - point3[0])
  y = abs(point1[1] - point3[1])
  alpha = -np.arctan2(y, x)
  if y > x:
    alpha = (math.pi / 2 + alpha)
  center = (((point1[0] + point2[0]) / 2), ((point3[1] + point4[1]) / 2))
  M = cv2.getRotationMatrix2D(center, np.degrees(alpha), 1.0)
  image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
  return image, alpha, center


def crop_image(point1, point2, point3, point4, image):
  x_min = min(point1[0], point2[0], point3[0], point4[0]) + 3
  x_max = max(point1[0], point2[0], point3[0], point4[0]) - 3
  y_min = min(point1[1], point2[1], point3[1], point4[1]) + 3
  y_max = max(point1[1], point2[1], point3[1], point4[1]) - 3
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


def canny_gray(image):
  edges = cv2.Canny(image, 100, 300, apertureSize=7)
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
  # display_image(image)
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


def find_figures(image):
  if photo is 1:
    image = cv2.medianBlur(image, 7)
    image = cv2.medianBlur(image, 7)
    img_bin = invert(dilate(erode(image_bin(image_gray(image)), 2)))
  else:
    img_bin = invert(dilate(erode(image_bin(image_gray(image)), 2)))

  shape = image.shape
  field_len = int(shape[0] / 8)

  img, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  regions_array = []
  hierarchy = hierarchy[0]
  for component in zip(contours, hierarchy):
    contour = component[0]
    hierarchy = component[1]
    if hierarchy[3] == -1:  # izdvoji samo skroz spoljne konture
      x, y, w, h = cv2.boundingRect(contour)
      area = cv2.contourArea(contour)
      if area > 100 and h < field_len and h > field_len / 2 and w > field_len / 3:
        region = img_bin[y:y + h + 1, x:x + w + 1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  regions_array = sorted(regions_array, key=lambda item: item[1][0])
  sorted_regions = [r[0] for r in regions_array]
  return image, sorted_regions


def find_model_figures(image):
  img_bin = invert(image_bin(image_gray(image)))

  field_len = 102

  img, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  regions_array = []
  hierarchy = hierarchy[0]
  for component in zip(contours, hierarchy):
    contour = component[0]
    hierarchy = component[1]
    if hierarchy[3] == -1:
      x, y, w, h = cv2.boundingRect(contour)
      area = cv2.contourArea(contour)
      if area > 100 and h > field_len / 2 and w > field_len / 3:
        region = img_bin[y:y + h + 1, x:x + w + 1]
        regions_array.append([resize_region(region), (x, y, w, h)])
  regions_array = sorted(regions_array, key=lambda item: item[1][0])
  sorted_regions = [r[0] for r in regions_array]

  return image, sorted_regions


def find_board_box(image_orig, image_bin):
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
  if photo is 1:
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  else:
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
  if len(sys.argv) == 2:
    main(sys.argv[1])
  else:
    print "train?"
