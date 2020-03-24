#from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import cv2
import os
import imutils
import numpy as np


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]


def load_from_folder(path):
    images = []
    for i, filename in enumerate(os.listdir(path)):
        # print(os.path.join(path, filename))
        img = cv2.imread(os.path.join(path, filename))
        images.append(process_image(img))
        if i == 49:
            break

    return images


def load_dataset(path):
    X = []
    y = []

    for id, gesture in enumerate(gestures):
        images = load_from_folder(os.path.join(path, gesture))
        X.extend(images)
        y.extend(len(images) * [id])

    return X, y


def image_to_vector(img):
    return imutils.resize(img, width=150).flatten()


dataset_path = os.path.join("..", "rock-paper-scissors")

gestures = ["paper", "rock", "scissors"]

train_path = os.path.join(dataset_path, "train")
train_images, train_ids = load_dataset(train_path)

train_X = np.array([image_to_vector(img) for img in train_images])
train_y = train_ids

# for x in train_X:
#     cv2.imshow("Test", x)
#     cv2.waitKey()

test_path = os.path.join(dataset_path, "test")
test_images, test_ids = load_dataset(test_path)

test_X = np.array([image_to_vector(img) for img in test_images])
test_y = test_ids



print(np.shape(train_X))
print(np.shape(train_y))

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(train_X, train_y)

result = knn.score(test_X, test_y)

print(result)

svc = SVC(C=1.0, kernel='rbf')
svc.fit(train_X, train_y)

result = svc.score(test_X, test_y)
print(result)