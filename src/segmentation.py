from typing import List, Tuple
import numpy as np
from config import *
import cv2 as cv
import math
import copy

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

    def __init__(self, pixels: List[Tuple[int, int]], image_height: int,
                 image_width: int):
        self.image_height = image_height
        self.image_width = image_width
        self.pixels = pixels
        self.bbox = self._calculate_bounding_box(image_height, image_width)
        self.local_image = self._build_local_image()
        self.W3 = self._calculate_w3(self._calculate_circumference(),
                                     len(self.pixels))
        # self.M1 =
        # self.M7 =

    def pixels_count(self) -> int:
        return len(self.pixels)

    def _calculate_bounding_box(
            self, image_height: int,
            image_width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        min_row, max_row, min_col, max_col = image_height, 0, image_width, 0
        for pixel in self.pixels:
            if pixel[0] < min_row:
                min_row = pixel[0]
            if pixel[0] > max_row:
                max_row = pixel[0]
            if pixel[1] < min_col:
                min_col = pixel[1]
            if pixel[1] > max_col:
                max_col = pixel[1]

        return (min_row, min_col), (max_row, max_col)

    def _calculate_circumference(self) -> int:
        circumference = 0
        local_height, local_width = self.local_image.shape
        for row in range(local_height):
            for col in range(local_width):
                if self.local_image[row][col] and self._has_neighbours(
                    (row, col)):
                    circumference += 1

        return circumference

    def _has_neighbours(self, current_pixel: Tuple[int, int]) -> bool:
        row, col = current_pixel
        return (self.local_image[row - 1][col] == 0
                or self.local_image[row][col - 1] == 0
                or self.local_image[row + 1][col] == 0
                or self.local_image[row][col + 1] == 0
                or self.local_image[row + 1][col - 1] == 0
                or self.local_image[row + 1][col + 1] == 0
                or self.local_image[row - 1][col - 1] == 0
                or self.local_image[row - 1][col + 1] == 0)

    def _calculate_w3(self, circumference: int, area: int) -> float:
        return (circumference / (2 * math.sqrt(math.pi * area))) - 1

    def _build_local_image(self) -> np.ndarray:
        local_height = self.bbox[1][0] - self.bbox[0][0] + 1
        local_width = self.bbox[1][1] - self.bbox[0][1] + 1
        # +2 to make sure that i have at least one black pixel at boarders
        local_image = np.zeros((local_height + 2, local_width + 2), dtype=bool)

        for pixel in self.pixels:
            # +1 to leave line of at least one black pixel on the left and on the top
            local_row = pixel[0] - self.bbox[0][0] + 1
            local_col = pixel[1] - self.bbox[0][1] + 1
            local_image[local_row][local_col] = True

        return local_image


def segmentation(image: np.ndarray) -> List[Segment]:
    image_height, image_width, _ = image.shape
    visited = np.zeros((image_height, image_width), dtype=bool)

    masks = []
    for color in color_dict.keys():
        masks.append(threshold(image, color_dict[color], color))

    for i, mask in enumerate(masks):
        cv.imshow(str(i), mask)
    # cv.waitKey(0)

    objects_list = []
    mask_objects_list = []
    for mask in masks:
        for row in range(image_height):
            for col in range(image_width):
                if not visited[row][col]:
                    # check if pixel is part of segment candidate (white color in mask)
                    if mask[row][col] != 0:
                        segment = flood_fill(mask, visited, (row, col))
                        if segment.pixels_count() > MIN_SEG_PIXEL_COUNT:
                            mask_objects_list.append(segment)
        objects_list.append(copy.deepcopy(mask_objects_list))
        mask_objects_list.clear()

    return objects_list


def flood_fill(image: np.ndarray, visited: np.ndarray,
               current_pixel: Tuple[int, int]) -> Segment:
    segment_pixels = []
    queue = [current_pixel]
    image_height, image_width = image.shape

    while queue:
        row, col = queue.pop()
        # check if visited
        if not visited[row][col]:
            visited[row][col] = True
            # check if pixel is part of segment candidate (white color in mask)
            if image[row][col] != 0:
                segment_pixels.append((row, col))
                # check neighbours
                if row - 1 >= 0:
                    queue.append((row - 1, col))
                if col - 1 >= 0:
                    queue.append((row, col - 1))
                if row + 1 < image_height:
                    queue.append((row + 1, col))
                if col + 1 < image_width:
                    queue.append((row, col + 1))

    return Segment(segment_pixels, image_height, image_width)


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
