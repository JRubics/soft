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
debug = 0
find_alpha = 0


def main(train, filename):
  if filename:
    print filename
    image_color = load_image('images/' + filename)
    display_image(image_color)
    image_color = dilate(erode(image_color, iterations=1))
    image_color = find_chessboard(image_color, image_color)
    selected_regions, regions = find_figures(image_color.copy())
    results = network(train, regions)
    output(results, selected_regions, regions)
  else:
    for filename in os.listdir('images'):
      print(filename)
      if filename != "other" and filename != "phone" and filename != "computer" and filename != "train" and filename != "screenshot" and filename != "main" and filename != "old":
        image_color = load_image('images/' + filename)
        display_image(image_color)
        image_color = dilate(erode(image_color, iterations=1))
        image_color = find_chessboard(image_color, image_color)
        selected_regions, regions = find_figures(image_color.copy())

        results = network(train, regions)
        output(results, selected_regions, regions)
        print("--------------------------------------------")


def output(results, selected_regions, regions):
  black = {}
  white = {}
  for idx, region in enumerate(regions):
    if find_color(region) is "crni":
      if figure_name(results[idx]) in black.keys():
        black[figure_name(results[idx])] = black[figure_name(results[idx])] + 1
      else:
        black[figure_name(results[idx])] = 1
    else:
      if figure_name(results[idx]) in white.keys():
        white[figure_name(results[idx])] = white[figure_name(results[idx])] + 1
      else:
        white[figure_name(results[idx])] = 1
  print "crni"
  print black
  print "beli"
  print white
  chess_winner(black, white)
  display_image(selected_regions)


def chess_winner(black, white):
  black_pawn = black["pijun"] if "pijun" in black.keys() else 0
  white_pawn = white["pijun"] if "pijun" in white.keys() else 0
  black_queen = black["kraljica"] if "kraljica" in black.keys() else 0
  black_rook = black["top"] if "top" in black.keys() else 0
  white_queen = white["kraljica"] if "kraljica" in white.keys() else 0
  white_rook = white["top"] if "top" in white.keys() else 0
  black_knight = black["konj"] if "konj" in black.keys() else 0
  black_bishop = black["lovac"] if "lovac" in black.keys() else 0
  white_knight = white["konj"] if "konj" in white.keys() else 0
  white_bishop = white["lovac"] if "lovac" in white.keys() else 0

  black_major = black_queen + black_rook
  black_minor = black_knight + black_bishop

  white_major = white_queen + white_rook
  white_minor = white_knight + white_bishop

  print "CRNI: kralj(1), major(" + str(black_major) + "), minor(" + str(black_minor) + "), pijuni(" + str(black_pawn) + ")"
  print "BELI: kralj(1), major(" + str(white_major) + "), minor(" + str(white_minor) + "), pijuni(" + str(white_pawn) + ")"
  if black_pawn + black_major + black_minor > white_pawn + white_major + white_minor:
    print "CRNI ima vise figura"
  elif black_pawn + black_major + black_minor < white_pawn + white_major + white_minor:
    print "BELI ima vise figura"
  else:
    print "JEDNAK BROJ FIGURA"
    if black_major > white_major:
      print "CRNI ima bolje figure"
    elif black_major < white_major:
      print "BELI ima bolje figure"
    else:
      if black_minor > white_minor:
        print "CRNI ima bolje figure"
      elif black_minor < white_minor:
        print "BELI ima bolje figure"
      else:
        print "SVE FIGURE SU ISTE"


def figure_data(result, region):
  return find_color(region) + " " + figure_name(result)


def find_color(image):
  rows, cols = image.shape
  b = 0
  w = 0
  for i in range(rows):
    for j in range(cols):
      if image[i, j] == 0:
        w += 1
      else:
        b += 1
  if b > w:
    return "crni"
  else:
    return "beli"


def figure_name(n):
  if n == 0:
    return "top"
  elif n == 1:
    return "konj"
  elif n == 2:
    return "lovac"
  elif n == 3:
    return "kraljica"
  elif n == 4:
    return "kralj"
  elif n == 5:
    return "pijun"


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

  result_vector = display_result(result, alphabet)
  return result_vector


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


def find_chessboard(orig, image):
  img = canny(image)
  if debug:
    display_image(img)
  img = erode(dilate(img, 2), 2)
  orig = find_board_box(image.copy(), img)
  return orig


def canny(image, threshold1=100, threshold2=300, apertureSize=3):
  gray = image_gray(image)
  edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)
  return dilate(edges)


def find_figures(image):
  img_bin = invert(dilate(erode(image_bin(image_gray(image)), 2)))
  if debug:
    display_image(img_bin)

  shape = image.shape
  field_len = int(shape[0] / 8)

  img, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  if debug:
    cpy = image.copy()
    cpy = cv2.drawContours(cpy, contours, -1, (0, 255, 0), 3)
    display_image(cpy)

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


def perspective_fix(image, contours):
  for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
      contour = approx
      break

  pts = contour.reshape(4, 2)
  rect = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  (tl, tr, br, bl) = rect
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  maxHeight = max(int(heightA), int(heightB))

  dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

  M = cv2.getPerspectiveTransform(rect, dst)
  warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  height, width, channels = warp.shape
  warp = warp[4:height - 4, 4:width - 4]
  if debug:
    display_image(warp)
  return warp


def find_board_box(image_orig, image_bin):
  img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
  return perspective_fix(image_orig, contours)


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


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(sys.argv[1], None)
  elif len(sys.argv) == 3:
    main(sys.argv[1], sys.argv[2])
  else:
    print "train?"
