# USAGE
# python svm_run.py -i dataset/test/ -v dataset/validation/ -t output/svm_h_proj.pickle -l output/le_h_proj.pickle -m h_proj

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import binarisation as bina
import features as feat


def run(input, trainedsvm, labelencoder, model) :

    if model == "pixel_nb" :
        print("[INFO] feature extractor model : pixel_number()")
        f_extractor = feat.pixel_number
    elif model == "h_proj" :
        print("[INFO] feature extractor model : horizontal_projection()")
        f_extractor = feat.horizontal_projection
    elif model == "v_proj" :
        print("[INFO] feature extractor model : vertical_projection()")
        f_extractor = feat.vertical_projection
    elif model == "native" :
        print("[INFO] feature extractor model : native()")
        f_extractor = feat.native
    else :
        print("[INFO] error : feature extractor model {} not found. Try pixel_nb, h_proj or v_proj".format(model))
        return

    print("[INFO] loading SVM...")
    recognizer = pickle.loads(open(trainedsvm, "rb").read())
    print("[INFO] loading labels...")
    le = pickle.loads(open(labelencoder, "rb").read())

    print("[INFO] extracting images paths...")
    imagePaths = list(paths.list_images(input))

    errors = []

    total = 0   

    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] testing image {}/{}...".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=50)

        imageBW = bina.binarise_lambda(image)

        features = f_extractor(imageBW)
        features = np.array(features).reshape(1, -1)

        preds = recognizer.predict_proba(features)[0]
        j = np.argmax(preds)
        proba = preds[j]
        sign = le.classes_[j]


        if sign != name :
            errors.append(imagePath)
            #print("        image : {}".format(imagePath))
            #print("        signe : {}".format(name))
            #print("        prediction : {}".format(sign))
            #print("        proba : {}%".format(round(proba * 100, 2)))

        total += 1

    return total, errors


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="the path to the input test directory")
    ap.add_argument("-v", "--validate", required=True, help="the path to the input validate directory")
    ap.add_argument("-t", "--trainedsvm", required=True, help="path to input model trained to recognize signs")
    ap.add_argument("-l", "--labelencoder", required=True, help="path to input label encoder")
    ap.add_argument("-m", "--model", required=True, help="type of features extracted (pixel_nb, h_proj, v_proj)")
    args = vars(ap.parse_args())

    total, errors = run(args["input"], args["trainedsvm"], args["labelencoder"], args["model"])
    v_total, v_errors = run(args["validate"], args["trainedsvm"], args["labelencoder"], args["model"])

    print("")
    print("[TEST] number of images tested : {}".format(total))
    print("[TEST] number of errors : {}".format(len(errors)))
    print("[TEST] taux de bonne prédiction : {}%".format(round(((total - len(errors)) / total) * 100, 2)))
    print("")
    print("[VALIDATE] number of images tested : {}".format(v_total))
    print("[VALIDATE] number of errors : {}".format(len(v_errors)))
    print("[VALIDATE] taux de bonne prédiction : {}%".format(round(((v_total - len(v_errors)) / v_total) * 100, 2)))
    #print("[RESULT] erreurs :")
    #print(errors)