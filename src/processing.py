import numpy as np
from typing import Tuple

BLUR_KERNEL_CLASSIC = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9],
                                [1 / 9, 1 / 9, 1 / 9]])
BLUR_KERNEL_GAUSS = np.array([[1 / 10, 1 / 10,
                               1 / 10], [1 / 10, 2 / 10, 1 / 10],
                              [1 / 10, 1 / 10, 1 / 10]])
BLUR_KERNEL_LAPLACE = np.array([[1 / 16, 2 / 16, 1 / 16],
                                [2 / 16, 4 / 16, 2 / 16],
                                [1 / 16, 2 / 16, 1 / 16]])
BLUR_KERNEL_CLASSIC_5x5 = np.full((5, 5), fill_value=1 / 25)


def applay_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_width, kernel_height = kernel.shape
    shift_width = int(np.floor(kernel_width / 2))
    shift_height = int(np.floor(kernel_height / 2))
    image_height, image_width, channels = image.shape
    output = np.zeros(image.shape, dtype=np.uint8)

    for row in range(shift_height, image_height - shift_height):
        for col in range(shift_width, image_width - shift_width):
            for ch in range(channels):
                output[row][col][ch] = np.sum(np.multiply(
                    image[:, :, ch][row - shift_width:row + shift_width + 1,
                                    col - shift_height:col + shift_height + 1],
                    kernel),
                                              dtype=np.uint8)
    return output


def ranking_filter(image: np.ndarray, kernel_shape: Tuple[int, int],
                   function) -> np.ndarray:
    image_height, image_width = image.shape
    shift_width = int(np.floor(kernel_shape[1] / 2))
    shift_height = int(np.floor(kernel_shape[0] / 2))
    output = np.zeros(image.shape, dtype=bool)

    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            output[row][col] = function(image[np.ix_(
                list(range(row - shift_width, row + shift_width + 1)),
                list(range(col - shift_height, col + shift_height + 1)))])

    return output


def optain_histogram(image: np.ndarray) -> np.ndarray:
    histogram = np.zeros(256, dtype=np.uint64)
    image_height, image_width, _ = image.shape

    for row in range(image_height):
        for col in range(image_width):
            histogram[image[row][col][2]] += 1

    return histogram


def create_LUT(histogram: np.ndarray, n_pixels: int) -> np.ndarray:
    LUT = np.zeros(len(histogram), dtype=np.uint64)
    p_sum = 0
    for i, hist_i in enumerate(histogram):
        p_sum += hist_i
        LUT[i] = p_sum * 255 / n_pixels

    return LUT


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    image_height, image_width, _ = image.shape

    histogram = optain_histogram(image)
    LUT = create_LUT(histogram, image_height * image_width)

    for row in range(image_height):
        for col in range(image_width):
            image[row][col][2] = np.uint8(LUT[image[row][col][2]])

    return image
