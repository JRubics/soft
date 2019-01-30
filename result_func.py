from base_func import *


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
