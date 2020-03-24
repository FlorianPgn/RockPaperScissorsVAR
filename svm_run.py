# USAGE
# python svm_run.py -i image.jpg -t output/svm_h.pickle -l output/le_svm_h.pickle -m h_proj

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import binarisation as bina
import features as feat


def load(trainedsvm, labelencoder, model):
    print("[INFO] loading SVM...")
    recognizer = pickle.loads(open(trainedsvm, "rb").read())

    print("[INFO] loading labels...")
    le = pickle.loads(open(labelencoder, "rb").read())
    return recognizer, le


def run(image, recognizer, le, model):

    if model == "pixel_nb" :
        print("[INFO] feature extractor model : pixel_number()")
        f_extractor = feat.pixel_number
    elif model == "h_proj" :
        print("[INFO] feature extractor model : horizontal_projection()")
        f_extractor = feat.horizontal_projection
    elif model == "v_proj" :
        print("[INFO] feature extractor model : vertical_projection()")
        f_extractor = feat.vertical_projection
    else :
        print("[INFO] error : feature extractor model {} not found. Try pixel_nb, h_proj or v_proj".format(model))
        return


    print("[INFO] feature extraction...")
    imageBW = bina.binarise_lambda(image)
    features = f_extractor(imageBW)
    features = features.reshape(1, -1)

    print("[INFO] prediction...")
    preds = recognizer.predict_proba(features)[0]
    j = np.argmax(preds)
    proba = preds[j]
    sign = le.classes_[j]

    return sign, proba


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="the path to the input image")
    ap.add_argument("-t", "--trainedsvm", required=True, help="path to input model trained to recognize signs")
    ap.add_argument("-l", "--labelencoder", required=True, help="path to input label encoder")
    ap.add_argument("-m", "--model", required=True, help="type of features extracted (pixel_nb, h_proj, v_proj)")
    args = vars(ap.parse_args())

    recognizer, le = load(args["trainedsvm"], args["labelencoder"])

    print("[INFO] loading image...")
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=100)

    prediction, proba = run(image, recognizer, le, args["model"])

    print("[RESULT] sign detected : {}".format(prediction))
    print("[RESULT] probability : {}%".format(round(proba * 100, 2)))