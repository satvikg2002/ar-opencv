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

# hough circle params
sc = 1      # scale
md = 30     # minimum distance between 2 circles
at = 40     # accumulator threshold, small numbers are more sensitive to false pos but make detection tolerant
    

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
    grid[h:2*h, 0:w] = np.dstack([cv2.Canny(grey, t / 2, t)] * 3)       # edge detection on greyscale
    grid[0:h, w:2*w] = np.dstack([blurred] * 3)                         # blurred grey image
    grid[h:2*h, w:2*w] = np.dstack([cv2.Canny(blurred, t / 2, t)] * 3)  # edge detection on blurred greyscale (reduces bg noise)

    # cv2.imshow("Image previews", grid)

    # hough gradients for circle detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, sc, md, t, at)

    if circles is not None and circles[0][0].ndim == 1:
        circle = circles[0][0]  # first circle
        x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
        print(x, y, radius)

        # Highlight the circle
        cv2.circle(image, (x, y), radius, (0, 0, 255), 2)
        # Draw dot in the center
        cv2.circle(image, (x, y), 1, (0, 0, 255), 3) 

    cv2.imshow('Image with detected circle', image)

    # press q to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break