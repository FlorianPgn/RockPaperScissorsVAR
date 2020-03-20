import cv2
import imutils
import numpy as np


thresh = lambda x: [255, 255, 255] if (132 < x[1] < 165) or (90 < x[2] < 105) else [0, 0, 0]

backSub = cv2.createBackgroundSubtractorMOG2(0)

images = ["images/paper.jpg", "images/scissors.jpg", "images/stone1.jpg", "images/stone2.jpg"]

im = cv2.imread("images/paper.jpg")
frame = imutils.resize(im, width=300)
(h, w) = frame.shape[:2]

# fgmask = backSub.apply(frame)
# kernel = np.ones((3, 3), np.uint8)
# fgmask = cv2.erode(fgmask, kernel, iterations=1)
# mask = cv2.bitwise_and(frame, frame, mask=fgmask)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

res = frame

# res = np.array([[thresh(col) for col in row] for row in frame], dtype=np.uint8)
# res = cv2.inRange(frame, (0, 132, 90), (255, 165, 105))
# res = cv2.flip(res*255, 1)

print(np.unique(res))

cv2.imshow("Result", res)

cv2.waitKey(0)
