import numpy as np


def sliding_window(img: np.ndarray, size: tuple, step: tuple):
    for y in range(0, img.shape[0] - size[0] + 1, step[0]):
        for x in range(0, img.shape[1] - size[1] + 1, step[1]):
            yield y, x, img[y: y + size[0], x: x + size[1]]
