import numpy as np
import math


def resize(original_img: np.ndarray, out_shape: tuple) -> np.ndarray:
    old_h, old_w = original_img.shape
    new_h, new_w = out_shape
    out_img = np.zeros((new_h, new_w))
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for row in range(new_h):
        for col in range(new_w):
            new_row = row * h_scale_factor
            new_col = col * w_scale_factor

            row_floor = np.floor(new_row)
            row_ceil = np.min(old_h - 1, math.ceil(new_row))
            col_floor = np.floor(new_col)
            col_ceil = np.min(old_w - 1, math.ceil(new_col))

            if (row_ceil == row_floor) and (col_ceil == col_floor):
                new_pixel = original_img[int(new_row), int(new_col)]
            elif row_ceil == row_floor:
                left_pixel = original_img[int(new_row), int(col_floor)]
                right_pixel = original_img[int(new_row), int(col_ceil)]
                new_pixel = left_pixel * (col_ceil - new_col) + right_pixel * (new_col - col_floor)
            elif col_ceil == col_floor:
                top_pixel = original_img[int(row_floor), int(new_col)]
                down_pixel = original_img[int(row_ceil), int(new_col)]
                new_pixel = (top_pixel * (row_ceil - new_row)) + (down_pixel * (new_row - row_floor))
            else:
                top_left_pixel = original_img[row_floor, col_floor]
                down_left_pixel = original_img[row_ceil, col_floor]
                top_right_pixel = original_img[row_floor, col_ceil]
                down_right_pixel = original_img[row_ceil, col_ceil]

                left_pixel = top_left_pixel * (row_ceil - new_row) + down_left_pixel * (new_row - row_floor)
                right_pixel = top_right_pixel * (row_ceil - new_row) + down_right_pixel * (new_row - row_floor)
                new_pixel = left_pixel * (col_ceil - new_col) + right_pixel * (new_col - col_floor)

            out_img[row, col] = new_pixel
    return out_img.astype(np.uint8)


def rgb2gray(img: np.ndarray):
    pass


def bgr2gray(img: np.ndarray):
    pass
