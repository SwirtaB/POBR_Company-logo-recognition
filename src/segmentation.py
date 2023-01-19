from typing import List, Tuple
import numpy as np
from config import *
import cv2 as cv

color_dict = {
    'red': [
        R_HUE_MIN_THRESHOLD, R_HUE_MAX_THRESHOLD, R_SATUR_MIN_THRSHOLD,
        R_SATUR_MAX_THRSHOLD, R_VALUE_MIN_THRSHOLD, R_VALUE_MAX_THRSHOLD
    ],
    'green': [
        G_HUE_MIN_THRESHOLD, G_HUE_MAX_THRESHOLD, G_SATUR_MIN_THRSHOLD,
        G_SATUR_MAX_THRSHOLD, G_VALUE_MIN_THRSHOLD, G_VALUE_MAX_THRSHOLD
    ],
    'blue': [
        B_HUE_MIN_THRESHOLD, B_HUE_MAX_THRESHOLD, B_SATUR_MIN_THRSHOLD,
        B_SATUR_MAX_THRSHOLD, B_VALUE_MIN_THRSHOLD, B_VALUE_MAX_THRSHOLD
    ],
    'yellow': [
        Y_HUE_MIN_THRESHOLD, Y_HUE_MAX_THRESHOLD, Y_SATUR_MIN_THRSHOLD,
        Y_SATUR_MAX_THRSHOLD, Y_VALUE_MIN_THRSHOLD, Y_VALUE_MAX_THRSHOLD
    ]
}


class Segment:

    def __init__(self, pixels: list):
        pass


def segmentation(image: np.ndarray) -> List[Segment]:
    image_height, image_width, _ = image.shape
    visited = np.zeros((image_height, image_width), dtype=np.uint8)

    masks = []
    for color in color_dict.keys():
        masks.append(threshold(image, color_dict[color], color))

    # for i, mask in enumerate(masks):
    #     cv.imshow(str(i), mask)

    objects_list = []
    for mask in masks:
        for row in range(image_height):
            for col in range(image_width):
                if not visited[row][col]:
                    segment = flood_fill(image)
                    if segment.pixels_count > MIN_SEG_PIXEL_COUNT:
                        objects_list.append(segment)

    return objects_list


def flood_fill(image: np.ndarray, visited: np.ndarray,
               current_pixel: Tuple[int, int]) -> Segment:
    pixels = [current_pixel]


def threshold(image: np.ndarray, thresholds: List[int],
              color: str) -> np.ndarray:
    image_height, image_width, _ = image.shape
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for row in range(image_height):
        for col in range(image_width):
            if color == 'red':
                mask[row][col] = _threshold_R(image[row][col], thresholds)
            else:
                mask[row][col] = _threshold(image[row][col], thresholds)

    return mask


def _delate():
    pass


def _erode():
    pass


def _threshold_R(pixel: np.ndarray, thresholds: List[int]) -> np.uint8:
    if (((0 <= pixel[0] <= thresholds[1]) or \
            (thresholds[0] <= pixel[0] <= 180)) and (\
        thresholds[2] <= pixel[1] <= thresholds[3]) and (\
        thresholds[4] <= pixel[2] <= thresholds[5])):
        return np.uint8(255)
    else:
        return np.uint8(0)


def _threshold(pixel: np.ndarray, thresholds: List[int]) -> np.uint8:
    if (thresholds[0] <= pixel[0] <= thresholds[1]) and (\
        thresholds[2] <= pixel[1] <= thresholds[3]) and (\
            thresholds[4] <= pixel[2] <= thresholds[5]):
        return np.uint8(255)
    else:
        return np.uint8(0)
