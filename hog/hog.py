import numpy as np


def _compute_histograms(magnitude, orientation, pixels_per_cell,
                        number_of_cells_columns, number_of_cells_rows, number_of_orientations):

    angle_span = 180. / number_of_orientations
    orientation_histogram = np.zeros((number_of_cells_rows, number_of_cells_columns, number_of_orientations))
    for i in range(number_of_orientations):
        orientation_start = angle_span * (i + 1)
        orientation_end = angle_span * i
        orientation_bitmap = (orientation >= orientation_end) & (orientation < orientation_start)
        curr_orientation_magnitude = magnitude * orientation_bitmap
        for row in range(number_of_cells_rows):
            for column in range(number_of_cells_columns):
                range_rows_stop_ = row * pixels_per_cell[0] + pixels_per_cell[0]
                range_rows_start_ = row * pixels_per_cell[0]
                range_columns_stop_ = column * pixels_per_cell[1] + pixels_per_cell[1]
                range_columns_start_ = column * pixels_per_cell[1]
                curr_cell_pixels = np.ogrid[range_rows_start_: range_rows_stop_, range_columns_start_:range_columns_stop_]
                cell_magnitude = curr_orientation_magnitude[tuple(curr_cell_pixels)]
                orientation_histogram[row, column, i] = cell_magnitude.sum() / (pixels_per_cell[0] * pixels_per_cell[1])
    return orientation_histogram


def _normalize_block(block): # L2-hys norm
    eps = 1e-5
    out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    out = np.minimum(out, 0.2)
    out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    return out


def _compute_gradients(image):
    rows_gradient = np.empty(image.shape, dtype=np.double)
    rows_gradient[0, :] = 0
    rows_gradient[-1, :] = 0
    rows_gradient[1:-1, :] = image[2:, :] - image[:-2, :]
    cols_gradient = np.empty(image.shape, dtype=np.double)
    cols_gradient[:, 0] = 0
    cols_gradient[:, -1] = 0
    cols_gradient[:, 1:-1] = image[:, 2:] - image[:, :-2]

    magnitude = np.hypot(cols_gradient, rows_gradient)
    orientation = np.rad2deg(np.arctan2(rows_gradient, cols_gradient)) % 180
    return magnitude, orientation


def hog(image: np.ndarray, orientations: int, pixels_per_cell: tuple, cells_per_block: tuple):  # only 2d nochannels gray
    n_cells_row = int(image.shape[0] // pixels_per_cell[0])
    n_cells_col = int(image.shape[1] // pixels_per_cell[1])

    magnitude, orientation = _compute_gradients(image)

    orientation_histogram = _compute_histograms(magnitude, orientation, pixels_per_cell, n_cells_col, n_cells_row, orientations)

    n_blocks_row = (n_cells_row - cells_per_block[0]) + 1
    n_blocks_col = (n_cells_col - cells_per_block[1]) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col, cells_per_block[0], cells_per_block[1], orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + cells_per_block[0], c:c + cells_per_block[1], :]
            normalized_blocks[r, c, :] = _normalize_block(block)

    return normalized_blocks.flatten()
