import os
import sys
from network_func import *
from chessboard_func import *
from result_func import *


def chess(train, filename, epochs=130):
  if filename:
    processing(train, filename, epochs)
  else:
    for filename in os.listdir('images'):
      if filename != "train" and filename != "main":
        processing(train, filename, epochs)
        print("--------------------------------------------")


def processing(train, filename, epochs):
  print filename
  image_color = load_image('images/' + filename)
  display_image(image_color)
  image_color = dilate(erode(image_color, iterations=1))
  image_color = find_chessboard(image_color, image_color)
  selected_regions, regions = find_figures(image_color.copy())
  results = network(train, regions, epochs)
  output(results, selected_regions, regions)


if __name__ == '__main__':
  if len(sys.argv) == 2:
    chess(sys.argv[1], None)
  elif len(sys.argv) == 3:
    chess(sys.argv[1], sys.argv[2])
  elif len(sys.argv) == 4:
    chess(sys.argv[1], sys.argv[2], int(sys.argv[3]))
  else:
    print "train?"
