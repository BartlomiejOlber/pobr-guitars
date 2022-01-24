from preprocessing.img_ops import nn_resize, bi_resize
import numpy as np


def _create_kernel(size=3, sig=2.):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def convolve(image: np.ndarray, kernel: np.ndarray, padding: int = 0, strides: int = 1):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    output = np.zeros((int(((image.shape[0] - kernel_h + 2 * padding) / strides) + 1),
                       int(((image.shape[1] - kernel_w + 2 * padding) / strides) + 1)))

    if padding != 0:
        image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        image_padded[padding:-padding, padding:-padding] = image
    else:
        image_padded = image

    for col in range(0, image.shape[1] - kernel_w + 1, strides):
        for row in range(0, image.shape[0] - kernel_h + 1, strides):
            output[row, col] = (kernel * image_padded[row: row + kernel_h, col: col + kernel_w]).sum()

    return output


def pyramid_gaussian(img: np.ndarray, downscale: float, max_layer: int):
    sigma = 2 * downscale / 6.0
    g_kernel = _create_kernel(3, sigma)
    layer = 0
    current_shape = img.shape

    prev_layer_image = img
    yield img

    while layer != max_layer:
        layer += 1
        out_shape = tuple([int(np.ceil(d / downscale)) for d in prev_layer_image.shape])

        smoothed = convolve(prev_layer_image, g_kernel)
        layer_image = nn_resize(smoothed, out_shape)

        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)

        if np.all(current_shape == prev_shape):
            break

        yield layer_image
