import argparse
import cv2
import numpy as np
import imutils

delta_YCrCb = 128       #-> 8-bit images
#delta_YCrCb = 32768     #-> 16 bits images
#delta_YCrCb = 0.5       #-> floating-point images

def manual_color_shift(image_BGR) :

    image_YCrCb = np.zeros(image_BGR.shape, np.uint8)

    for x in range(image_YCrCb.shape[0]) :

        for y in range(image_YCrCb.shape[1]) :

            Y = 0.299 * image_BGR[x,y,2] + 0.587 * image_BGR[x,y,1] + 0.114 * image_BGR[x,y,0]
            Cr = (image_BGR[x,y,2] - Y) * 0.713 + delta_YCrCb
            Cb = (image_BGR[x,y,0] - Y) * 0.564 + delta_YCrCb

            image_YCrCb[x,y] = (Y, Cr, Cb)
    
    return image_YCrCb

def binarise_lambda(image_BGR) :
    image_YCrCb = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2YCR_CB)

    threshold = 18
    #Rcr = [130:165]
    Cr_min = 130 - threshold
    Cr_max = 165 + threshold
    #Rcb = [80:105]
    Cb_min = 80 - threshold
    Cb_max = 105 + threshold

    #bina = lambda x : [255, 255, 255] if (x[1] in range(Cr_min, Cr_max + 1) and x[2] in range(Cb_min, Cb_max + 1))  else [0, 0, 0]
    bina = lambda x : [255, 255, 255] if (x[1] >= Cr_min and x[1] <= Cr_max and x[2] >= Cb_min and x[2] <= Cb_max)  else [0, 0, 0]

    imageBW = np.array([[bina(pixel) for pixel in col] for col in image_YCrCb], dtype = np.uint8)

    return imageBW


def binarise(image_BGR) :

    image_YCrCb = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2YCR_CB)
    #cv2.imshow("image_YCrCb", image_YCrCb)
    #cv2.waitKey(0)

 
    #threshold = 18
    threshold = 0
    #Rcr = [130:165]
    Cr_min = 130
    Cr_max = 165
    #Rcb = [80:105]
    Cb_min = 80
    Cb_max = 105

    image_BW = image_YCrCb
    #image_BW = image_BW + 255

    for i, row in enumerate(image_YCrCb) :

        for j, pixel in enumerate(row) :
            Crxy, Cbxy = (pixel[1], pixel[2])
            #print(image_YCrCb[x, y])
            if (Crxy in range(Cr_min, Cr_max + 1)) & (Cbxy in range(Cb_min, Cb_max + 1)) :
                image_BW[i, j] = 255
            else :
                image_BW[i, j] = 0
                #D1 = distance_euclidienne((Crxy, Cbxy), (Cr_min, Cb_min))
                #D2 = distance_euclidienne((Crxy, Cbxy), (Cr_min, Cb_max))
                #D3 = distance_euclidienne((Crxy, Cbxy), (Cr_max, Cb_min))
                #D4 = distance_euclidienne((Crxy, Cbxy), (Cr_max, Cb_max))

                #if (D1 <= threshold) | (D2 <= threshold) | (D3 <= threshold) | (D4 <= threshold) :
                #    image_BW[i, j] = 255



    return image_BW

def distance_euclidienne(vector_2D_1, vector_2D_2) :

    return np.linalg.norm(np.array(vector_2D_1)-np.array(vector_2D_2))
    #return sqrt((vector_2D_1[0] - vector_2D_2[0]) ^ 2 + (vector_2D_1[1] - vector_2D_2[1]) ^ 2)



###

def main(image_path):

    image = cv2.imread(image_path) #attention, BGR not RGB 
    image = imutils.resize(image, width=100)
    imageBW = binarise_lambda(image)
    cv2.imshow("Image", image)
    cv2.imshow("Image2", imageBW)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="the path to the image")
    args = vars(ap.parse_args())
    main(args["image"])
