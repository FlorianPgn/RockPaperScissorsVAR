# USAGE
# python svm_extract.py -i dataset/train/ -o output/features_h_proj.pickle -m h_proj -s 100

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import features as feat
import binarisation as bina


def extract(input, output, model, size):
    
    if model == "pixel_nb" :
        print("[INFO] feature extractor model : pixel_number()")
        f_extractor = feat.pixel_number
    elif model == "h_proj" :
        print("[INFO] feature extractor model : horizontal_projection()")
        f_extractor = feat.horizontal_projection
    elif model == "v_proj" :
        print("[INFO] feature extractor model : vertical_projection()")
        f_extractor = feat.vertical_projection
    elif model == "d_proj" :
        print("[INFO] feature extractor model : dual_projection()")
        f_extractor = feat.dual_projection
    elif model == "native" :
        print("[INFO] feature extractor model : native()")
        f_extractor = feat.native
    else :
        print("[INFO] error : feature extractor model {} not found. Try pixel_nb, h_proj, v_proj, d_proj or native".format(model))
        return
    

    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(input))

    knownFeatures = []
    knownNames = []

    total = 0   

    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=size)
        imageBW = bina.binarise_lambda(image)

        feature = f_extractor(imageBW)

        knownNames.append(name)
        knownFeatures.append(feature)
        total += 1

    print("[INFO] gathering features and labels...")
    data = {"features": knownFeatures, "labels": knownNames}
    #print(data)

    print("[INFO] saving features and labels...")
    f = open(output, "wb")
    f.write(pickle.dumps(data))
    f.close()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input directory of hands")
    ap.add_argument("-o", "--output", required=True, help="path to output file")
    ap.add_argument("-m", "--model", required=True, help="type of features extracted (pixel_nb, h_proj, v_proj)")
    ap.add_argument("-s", "--size", type=int, default=100, help="width for the image resizing")
    args = vars(ap.parse_args())

    extract(args["input"], args["output"], args["model"], args["size"])