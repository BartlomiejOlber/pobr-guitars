import numpy as np


def nn_resize(img: np.ndarray, out_shape: tuple) -> np.ndarray:
    new_width, new_height = out_shape[1], out_shape[0]
    nearest_id = np.ogrid[0:1, 0:1]
    if img.ndim == 3:
        height, width, channels = img.shape
        new = np.ones((new_height, new_width, channels), dtype=img.dtype)
    else:
        height, width = img.shape
        new = np.ones((new_height, new_width))
    scale_x = new_width / width
    scale_y = new_height / height
    for y in range(new_height):
        for x in range(new_width):
            nearest_id[0][0] = int(np.round(y / scale_y))
            nearest_id[1][0] = int(np.round(x / scale_x))
            new[y, x] = img[tuple(nearest_id)]
    return new


def bi_resize(img: np.ndarray, out_shape: tuple) -> np.ndarray:
    old_h, old_w = img.shape
    new_h, new_w = out_shape
    out_img = np.zeros((new_h, new_w))
    w_scale_factor = old_w / new_w if new_h != 0 else 0
    h_scale_factor = old_h / new_h if new_w != 0 else 0
    for y in range(new_h):
        for x in range(new_w):
            exact_new_x = x / w_scale_factor
            exact_new_y = y / h_scale_factor

            x_floor = np.min(int(np.floor(exact_new_x)), old_w - 1)
            y_floor = np.min(int(np.floor(exact_new_y)), old_h - 1)
            x_ceil = np.min(int(np.ceil(exact_new_x)), old_w - 1)
            y_ceil = np.min(int(np.ceil(exact_new_y)), old_h - 1)

            top_left_pixel = img[y_floor, x_floor]
            top_right_pixel = img[y_floor, x_ceil]
            down_left_pixel = img[y_ceil, x_floor]
            down_right_pixel = img[y_ceil, x_ceil]

            interpolated_top_pixel = (x_ceil - exact_new_x) * top_left_pixel + (exact_new_x - x_floor) * top_right_pixel
            interpolated_down_pixel = (x_ceil - exact_new_x) * down_left_pixel + (
                        exact_new_x - x_floor) * down_right_pixel

            new_pixel = (y_ceil - exact_new_y) * interpolated_top_pixel + (
                        exact_new_y - y_floor) * interpolated_down_pixel
            out_img[y, x] = np.round(new_pixel)

    return out_img


def rgb2gray(img: np.ndarray):
    pass


def bgr2gray(img: np.ndarray):
    pass
