from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import svm_run

DEBUG = True


def debug(value):
    if DEBUG:
        print(value)


def compute_properties(img, display=True):
    # calculate moments of binary image
    M = cv2.moments(img)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = cY = 0

    # Find contours for bounding box
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # find the largest countour by area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    else:
        x = y = w = h = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if display:
        # draw point at the centroid
        cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        # draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img, (cX, cY), (x, y, w, h)


def ycbcr_binarize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cr = cv2.inRange(channels[1], crMin, crMax)
    cb = cv2.inRange(channels[2], cbMin, cbMax)
    return cv2.bitwise_or(cr, cb)


def remove_bg(img, bg, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, bg)
    return cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

backSub = cv2.createBackgroundSubtractorMOG2(0)
backgroundSet = False

t = 25
cbMin = 90
cbMax = 105
crMin = 134
crMax = 165

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio)
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    if backgroundSet:
        # Backgroud substraction technique
        # fgmask = backSub.apply(frame)
        # kernel = np.ones((3, 3), np.uint8)
        # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        # fgmask = cv2.erode(fgmask, kernel, iterations=1)
        # masked_img = cv2.bitwise_and(frame, frame, mask=fgmask)

        # YCbCr technique
        thresh = ycbcr_binarize(frame)

        # Background substraction with firt frame
        # thresh = remove_bg(frame, bg, t)
        # Try to eliminate most of the noise
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=3)

        # cv2.imshow("Result", thresh)
        # Split left and right (one player per side a.k.a 1 hand per side)
        res = cv2.flip(thresh, 1)
        left = res[:, :int(w / 2)]
        right = res[:, int(w / 2):]

        # Compute and display visual properties
        left, l_centroid, l_bounds = compute_properties(left)
        right, r_centroid, r_bounds = compute_properties(right)

        # Attach back left and right with a separator in the middle
        final = np.hstack((left, np.full((h, 1, 3), 125, dtype=np.uint8), right))

        cv2.imshow("Result", final)

    else:
        res = cv2.flip(frame, 1)
        cv2.imshow("Normal", res)

    # update the FPS counter
    fps.update()
    # show the output frame
    # cv2.imshow("Frame", flip)
    key = cv2.waitKey(1)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if key == ord("b"):
        # backSub = cv2.createBackgroundSubtractorMOG2(0)
        # bg = frame
        bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
        backgroundSet = True
    if key == ord("a"):
        crMin -= 1
    if key == ord("z"):
        crMin += 1
    if key == ord("e"):
        crMax -= 1
    if key == ord("r"):
        crMax += 1
    if key == ord("w"):
        cbMin -= 1
    if key == ord("x"):
        cbMin += 1
    if key == ord("c"):
        cbMax -= 1
    if key == ord("v"):
        cbMax += 1
    if key == ord("p"):
        print("Cr : [{}, {}] - Cb : [{}, {}]".format(crMin, crMax, cbMin, cbMax))
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
