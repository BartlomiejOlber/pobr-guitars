import cv2
import joblib
import numpy as np
from preprocessing import img_ops
from detection import pyramid, util, window
from hog import hog


def visualize():
    counter2 = 0
    for (y1, x1, y2, x2) in bboxes:
        if counter2 == max_count:
            break

        cv2.rectangle(clone, (x1, y1), (x2, y2), green, bbox_thickness)
        cv2.putText(clone, 'Guitar', (x1 - 2, y1 - 2), 1, 0.75, black, bbox_thickness)
        counter2 += 1


if __name__ == "__main__":
    images = ['images/test/test_guitars_2_2.jpg', 'images/test/acdc5.jpeg', "images/test/pink.jpeg", 'images/test/test_guitars_2.png',
              'images/test/acdc3.jpeg', "images/test/guns.jpeg", 'images/test/metallica.jpeg']
    size = (48, 48)
    max_img_size = 400 * 256
    step_size = (9, 9)
    downscale = 1.5
    green = (0, 255, 0)
    black = (0, 0, 0)
    bbox_thickness = 2
    max_count = 2
    overlap_threshold = .1
    model = joblib.load('bin/svc.pt')
    for img_path in images:

        detections = []
        scale_counter = 0
        image = cv2.imread(str(img_path))
        img_gray = cv2.imread(str(img_path), 0)
        if image is None:
            continue
        if img_gray.size > max_img_size:
            img_gray = img_ops.nn_resize(img_gray, (256, 400))
            image = cv2.resize(image, (400, 256))

        for im_scaled in pyramid.pyramid_gaussian(img_gray, downscale=downscale, max_layer=4):
            if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
                break
            for (y, x, current_window) in window.sliding_window(im_scaled, size, step_size):
                if current_window.shape[0] != size[1] or current_window.shape[1] != size[0]:
                    continue

                img_hog = hog.hog(current_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
                img_hog = img_hog.reshape(1, -1)
                pred = model.predict(img_hog)
                if pred == 1:
                    if model.decision_function(img_hog) > 1.6:
                        detections.append(
                            (int(y * (downscale ** scale_counter)), int(x * (downscale ** scale_counter)), model.decision_function(img_hog),
                             int(size[0] * (downscale ** scale_counter)),
                             int(size[1] * (downscale ** scale_counter))))

            scale_counter += 1
        clone = image.copy()
        bboxes = np.array([[y, x, y + h_offset, x + w_offset] for (y, x, _, h_offset, w_offset) in detections])
        scores = np.array([score[0] for (_, _, score, _, _) in detections])
        bboxes = util.select_best(bboxes, scores=scores, threshold=overlap_threshold)
        visualize(bboxes, max_count)
        cv2.imshow(str(img_path), clone)

    cv2.waitKey(0)
