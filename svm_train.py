
# USAGE
# python svm_train.py -d output/features_h_proj.pickle -t output/svm_h_proj.pickle -l output/le_h_proj.pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

def train(trainingdatas, trainedsvm, labelencoder) :

    print("[INFO] loading training datas...")
    data = pickle.loads(open(trainingdatas, "rb").read())

    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["labels"])

    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="precomputed", probability=True)
    recognizer.fit(data["features"], labels)

    print("[INFO] saving the trained SVM...")
    f = open(trainedsvm, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    print("[INFO] saving the label encoder...")
    f = open(labelencoder, "wb")
    f.write(pickle.dumps(le))
    f.close()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "---trainingdatas", required=True, help="path to extracted data features")
    ap.add_argument("-t", "--trainedsvm", required=True, help="path to output model trained to recognize signs")
    ap.add_argument("-l", "--labelencoder", required=True, help="path to output label encoder")
    args = vars(ap.parse_args())

    train(args["trainingdatas"], args["trainedsvm"], args["labelencoder"])