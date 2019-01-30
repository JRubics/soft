from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import model_from_json

import numpy as np
from base_func import *


find_alpha = 0


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
