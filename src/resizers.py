from typing import Callable
import numpy as np


def resize(image: np.ndarray, x_size: np.uint, y_size: np.uint,
           algorithm: Callable) -> np.ndarray:
    image_y, image_x, _ = image.shape
    scale_x = x_size / image_x
    scale_y = y_size / image_y
    scaled_image = np.zeros((y_size, x_size, 3), dtype=np.uint8)

    for y in range(y_size):
        for x in range(x_size):
            if algorithm.__name__ == "nn_interpolation":
                scaled_image[y][x] = algorithm(image, x, y, scale_x, scale_y)
            elif algorithm.__name__ == "bilinear_interpolation":
                scaled_image[y][x] = algorithm(image, x, y, scale_x, scale_y,
                                               image_x, image_y)

    return scaled_image


def nn_interpolation(image: np.ndarray, x: np.uint, y: np.uint,
                     scale_x: np.uint, scale_y: np.uint) -> np.ndarray:
    return image[np.uint(np.round(y / scale_y))][np.uint(np.round(x /
                                                                  scale_x))]


def bilinear_interpolation(image: np.ndarray, x: np.uint, y: np.uint,
                           scale_x: np.uint, scale_y: np.uint,
                           image_x: np.uint, image_y: np.uint) -> np.ndarray:
    #coordinates in original image
    x_ori = x / scale_x
    y_ori = y / scale_y

    #coordinates of neighboring points
    x1 = np.uint(np.floor(x_ori))
    y1 = np.uint(np.floor(y_ori))
    x2 = min(np.uint(np.ceil(x_ori)), image_x - 1)
    y2 = min(np.uint(np.ceil(y_ori)), image_y - 1)

    if x1 == x2 and y1 == y2:
        return image[np.uint(y_ori)][np.uint(x_ori)]

    #interpolating p1, p2 and p
    if x1 == x2:
        p1 = image[y1][x1]
        p2 = image[y2][x2]

        p = (y2 - y_ori) * p1 + (y_ori - y1) * p2
    elif y1 == y2:
        p1 = image[y1][x1]
        p2 = image[y1][x1]

        p = (x2 - x_ori) * p1 + (x_ori - x1) * p2
    else:
        q11 = image[y1][x1]
        q12 = image[y1][x2]
        q21 = image[y2][x1]
        q22 = image[y2][x2]

        p1 = (x2 - x_ori) * q11 + (x_ori - x1) * q12
        p2 = (x2 - x_ori) * q21 + (x_ori - x1) * q22
        p = (y2 - y_ori) * p1 + (y_ori - y1) * p2

    return p.astype(np.uint8)