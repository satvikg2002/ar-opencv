# imports
import cv2
import numpy as np
import time
import math

# built in camera
capture = cv2.VideoCapture(0)
print(capture.get(cv2.CAP_PROP_FPS))

t = 100     # threshold of canny edge detector
w = 640     # width of image for uniformity

while True:
    ret, image = capture.read()

    img_height, img_width, _ = image.shape
    scale = w // img_width
    h = img_height * scale
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)

    # filters
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(grey, 15)

    # 2x2 grid with all previews
    grid = np.zeros([2*h, 2*w, 3], np.uint8)
    grid[0:h, 0:w] = image

    # rgb to greyscale 8bit
    grid[h:2*h, 0:w] = np.dstack([cv2.Canny(grey, t / 2, t)] * 3)
    grid[0:h, w:2*w] = np.dstack([blurred] * 3)
    grid[h:2*h, w:2*w] = np.dstack([cv2.Canny(blurred, t / 2, t)] * 3)

    cv2.imshow("Image previews", grid)

    # press q to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break