import numpy as np


def sliding_window(img: np.ndarray, size: tuple, step: tuple):
    for y in range(0, img.shape[0], step[1]):
        for x in range(0, img.shape[1], step[0]):
            yield y, x, img[y: y + size[1], x: x + size[0]]
