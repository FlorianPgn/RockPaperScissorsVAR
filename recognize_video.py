from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

im = cv2.imread("images/stone1.jpg")

# start the FPS throughput estimator
fps = FPS().start()

extract = lambda x: [255, 255, 255] if (132 < x[1] < 165) or (90 < x[2] < 105) else [0, 0, 0]

backSub = cv2.createBackgroundSubtractorMOG2(0)
backgroundSet = False

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    if backgroundSet:

        fgmask = backSub.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        masked_img = cv2.bitwise_and(frame, frame, mask=fgmask)

        ycrcb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cr = cv2.inRange(channels[1], 132, 165)
        cb = cv2.inRange(channels[2], 90, 105)
        thresh = cv2.bitwise_or(cr, cb)

        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply thresholding to eliminate noise
        # thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)
        thresh = cv2.erode(thresh, None, iterations=1)

        # res = masked_img

        # print(np.unique(res))
        # print(np.shape(res))
        # print(res)

        res = cv2.flip(thresh, 1)
        # print(np.shape(res))
        # cv2.imshow("BW", frame)
    else:
        res = frame
    cv2.imshow("Mask", res)

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

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
