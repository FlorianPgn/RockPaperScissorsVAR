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
import scipy.interpolate as si

DEBUG = False


def debug(value):
    if DEBUG:
        print(value)


def smooth(c):
    x, y = c.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # print(x, y)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = si.splprep([x, y], u=None, s=1.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 25)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = si.splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    return np.asarray(res_array, dtype=np.int32)


def compute_properties(img, display=True):
    # Find contours for bounding box and convex hull
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # find the largest countour by area
        widestC = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(widestC)
        hull = cv2.convexHull(widestC)
    else:
        x = y = w = h = 0
        widestC = hull = None

    box = get_region(img, (x, y, w, h))

    # calculate moments of binary image
    M = cv2.moments(box)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = cY = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if display:
        # draw point at the centroid
        cv2.circle(img, (x+cX, y+cY), 5, (0, 0, 255), -1)
        # draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if hull is not None:
            cv2.drawContours(img, [hull], 0, (0, 0, 255))

    return img, (x+cX, y+cY), (x, y, w, h), widestC, hull


def get_coord(c, idx):
    return np.array(c[idx][0])


def count_defects(c, img):
    if c is None:
        return 0
    count = 0
    hull = cv2.convexHull(c, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(c, hull)
        if defects is not None:
            for d in defects:
                start_idx, end_idx, farthest_idx, _ = d[0]
                start = get_coord(c, start_idx)
                end = get_coord(c, end_idx)
                farthest = get_coord(c, farthest_idx)
                # cv2.circle(img, (farthest[0], farthest[1]), 5, (255, 0, 0), 2)
                if get_angle(farthest, start, end) < 1.6:
                    count += 1

    return count


# Return angle of p1p2 and p1p3
def get_angle(p1, p2, p3):
    dist_a = np.linalg.norm(p2-p1)
    dist_b = np.linalg.norm(p3-p1)
    dist_c = np.linalg.norm(p3-p2)
    x = (dist_a ** 2 + dist_b ** 2 - dist_c ** 2) / (2 * dist_a * dist_b)

    return np.arccos(x)


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


def get_region(img, box):
    x, y, w, h = box

    (height, width) = img.shape[:2]
    if h < height and w < width:
        return img[y:y+h, x:x+w]
    else:
        debug("False region")
        return img


def write_text(img, text, position, color=(0, 0, 255), center=True, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, line_type=1):
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_type)[0]
    if center:
        position = (int(position[0]-text_width/2), int(position[1]-text_height/2))
    cv2.putText(img, text, position, font, font_scale, color, line_type)


def get_shape_from_count(count):
    shape = "Paper" if count > 3 else "Scissors"
    return "Rock" if count == 0 else shape


def reset_bg():
    back_sub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=False)
    bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    background_set = True
    cv2.imshow("Bg", bg)
    return back_sub, bg, background_set



# ---- MAIN -----

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

back_sub = cv2.createBackgroundSubtractorMOG2(0)
background_set = False

t = 30
cbMin = 90
cbMax = 105
crMin = 134
crMax = 165

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio)
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]

    if background_set:
        # YCbCr technique to keep global skin zones
        thresh = ycbcr_binarize(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        thresh = cv2.erode(thresh, kernel, iterations=3)
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        # Mask the frame and keep skin zones
        frame = cv2.bitwise_and(frame, frame, mask=thresh)

        # cv2.imshow("Skin mask", frame)

        # Backgroud substraction technique
        # Background substraction with first frame
        # fgmask = remove_bg(frame, bg, t)
        fgmask = back_sub.apply(frame, learningRate=0000)
        # Smooth mask by blurring it and thresholding it
        fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

        thresh = fgmask
        # Try to eliminate most of the noise
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        frame = cv2.bitwise_and(frame, frame, mask=fgmask)

        # cv2.imshow("Mask", thresh)
        # cv2.imshow("Masked live", frame)

        # Split left and right (one player per side a.k.a 1 hand per side)
        res = cv2.flip(thresh, 1)
        left = res[:, :int(w / 2)]
        right = res[:, int(w / 2):]

        # Compute and display visual properties
        left, l_centroid, l_bounds, l_cnt, l_hull = compute_properties(left)
        right, r_centroid, r_bounds, r_cnt, r_hull = compute_properties(right)

        l_finger_count = count_defects(l_cnt, left)
        r_finger_count = count_defects(r_cnt, right)

        left_hand = get_region(left, l_bounds)
        if np.shape(left_hand)[0] > 0 and np.shape(left_hand)[1] > 0:
            # cv2.imshow("Left hand", left_hand)
            pass

        # Attach back left and right with a separator in the middle
        final = np.hstack((left, np.full((h, 1, 3), 125, dtype=np.uint8), right))

        # Display Cr Cb intervals
        text = "Cr: [{}, {}] - Cb: [{}, {}]".format(crMin, crMax, cbMin, cbMax)
        write_text(final, text, (w/2, 30))

        # Display left hand detection
        write_text(final, str(l_finger_count), (40, 30), color=(255, 0, 255))
        write_text(final, get_shape_from_count(l_finger_count), (40, 60), color=(255, 0, 255))

        # Display right hand detection
        write_text(final, str(r_finger_count), (w-40, 30), color=(255, 0, 255))
        write_text(final, get_shape_from_count(r_finger_count), (w-40, 60), color=(255, 0, 255))

        cv2.imshow("Result", final)

    else:
        res = cv2.flip(frame, 1)
        cv2.imshow("Normal", res)

    # update the FPS counter
    fps.update()
    # show the output frame
    key = cv2.waitKey(1)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    reset = False
    if key == ord("k"):
        t -= 5
    if key == ord("l"):
        t += 5

    if key == ord("a"):
        crMin -= 1
        reset = True
    if key == ord("z"):
        crMin += 1
        reset = True
    if key == ord("e"):
        crMax -= 1
        reset = True
    if key == ord("r"):
        crMax += 1
        reset = True
    if key == ord("w"):
        cbMin -= 1
        reset = True
    if key == ord("x"):
        cbMin += 1
        reset = True
    if key == ord("c"):
        cbMax -= 1
        reset = True
    if key == ord("v"):
        cbMax += 1
        reset = True

    if key == ord("b") or reset:
        back_sub, bg, background_set = reset_bg()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
