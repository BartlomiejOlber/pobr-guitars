from hog import hog
from preprocessing import img_ops, img_ops
import time
import joblib, cv2, pathlib

from sklearn.svm import LinearSVC
import numpy as np

if __name__ == "__main__":
    train_data = []
    train_labels = []
    pos_imgs_path = 'images/train/positive/'
    neg_imgs_path = 'images/train/negative/'
    model_path = 'bin/svc.pt'

    start_time = time.time()

    for filename in pathlib.Path(pos_imgs_path).rglob("*.*g"):
        img = cv2.imread(str(filename), 0)
        img = img_ops.nn_resize(img, (48, 48))
        img_hog = hog.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
        train_data.append(img_hog)
        train_labels.append(1.)

    positive_count = len(train_labels)
    train_data_ = []

    for i, filename in enumerate(pathlib.Path(neg_imgs_path).rglob("*g")):
        img = cv2.imread(str(filename), 0)
        img = img_ops.nn_resize(img, (48, 48))
        img_hog = hog.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
        train_data_.append(img_hog)
        train_labels.append(0.)

    train_data = train_data + train_data_
    train_data = np.float32(train_data)
    train_labels = np.array(train_labels)

    model = LinearSVC(verbose=1, max_iter=2000)
    model.fit(train_data, train_labels)
    joblib.dump(model, model_path)
    print("--- %s seconds ---" % (time.time() - start_time))
