
import binarisation as b
import argparse
import cv2
import numpy as np
import imutils

#shape : (hauteur, largeur, (couleur))

def pixel_number(imageBW) :
    return int(np.sum(imageBW == [255, 255, 255]) / 3)


def vertical_projection(imageBW) :
    projection = np.zeros((imageBW.shape[0]), dtype = np.uint8)
    for i, row in enumerate(imageBW) :
        projection[i] = np.sum(row == [255, 255, 255]) / 3
    return projection

def horizontal_projection(imageBW) :
    image_rotate = cv2.rotate(imageBW, cv2.ROTATE_90_CLOCKWISE)
    return vertical_projection(image_rotate)

def native(imageBW) :
    #img = np.zeros(([imageBW.shape[0], imageBW.shape[1]]), dtype = np.uint8)
    #for i, row in enumerate(imageBW) :
    #    for j, pixel in enumerate(row) :
    #        if np.sum(pixel == [255, 255, 255]) / 3 == 255 :
    #            img[i][j] = 1

    image_flat = list()
    for row in range(0, imageBW.shape[0]) :
        for pixel in range(0, imageBW.shape[1]) :
            if imageBW[row][pixel][0] == 255 :
                image_flat.append(1)
            else :
                image_flat.append(0)


    return image_flat

def main(image_path):

    image = cv2.imread(image_path) #attention, BGR not RGB 
    image = imutils.resize(image, width=100)
    imageBW = b.binarise_lambda(image)
    print(pixel_number(imageBW))
    print(vertical_projection(imageBW))
    print(horizontal_projection(imageBW))
    cv2.imshow("Image2", imageBW)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="the path to the image")
    args = vars(ap.parse_args())
    main(args["image"])