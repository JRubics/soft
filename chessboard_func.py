from base_func import *


debug = 0


def find_chessboard(orig, image):
  img = canny(image)
  if debug:
    display_image(img)
  img = erode(dilate(img, 2), 2)
  orig = find_contours(image.copy(), img)
  return orig


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


def find_contours(image_orig, image_bin):
  img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
  return perspective_fix(image_orig, contours)
