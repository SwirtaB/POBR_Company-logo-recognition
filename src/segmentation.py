from typing import List, Tuple
import numpy as np
from config import *
from processing import ranking_filter
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
        self.pixels = pixels
        self.bbox = self._calculate_bounding_box(image_height, image_width)
        self.global_center = ((self.bbox[1][0] + self.bbox[0][0]) / 2,
                              (self.bbox[1][1] + self.bbox[0][1]) / 2)
        self.local_image = self._build_local_image()
        self.center = (round(self._calculate_m_pq(1, 0) / self.pixels_count()),
                       round(self._calculate_m_pq(0, 1) / self.pixels_count()))
        self.W3 = self._calculate_w3(self._calculate_circumference(),
                                     len(self.pixels))
        self.M1_norm = self._calculate_M1_normalized()
        self.M7_norm = self._calculate_M7_normalized()

    def pixels_count(self) -> int:
        return len(self.pixels)

    def update_image(self, image: np.ndarray):
        self.local_image = image
        self.center = (round(self._calculate_m_pq(1, 0) / self.pixels_count()),
                       round(self._calculate_m_pq(0, 1) / self.pixels_count()))
        self.W3 = self._calculate_w3(self._calculate_circumference(),
                                     len(self.pixels))
        self.M1_norm = self._calculate_M1_normalized()
        self.M7_norm = self._calculate_M7_normalized()

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

    def _calculate_m_pq(self, p: int, q: int) -> int:
        local_height, local_width = self.local_image.shape
        m_pq = 0
        for row in range(local_height):
            for col in range(local_width):
                # local image has bool value - True if there is object's pixel, False if background
                if self.local_image[row][col]:
                    m_pq += pow(row, p) * pow(col, q)
        return m_pq

    def _calculate_M_pq(self, p: int, q: int) -> int:
        local_height, local_width = self.local_image.shape
        M_pq = 0
        for row in range(local_height):
            for col in range(local_width):
                # local image has bool value - True if there is object's pixel, False if background
                if self.local_image[row][col]:
                    M_pq += pow(row - self.center[0], p) * pow(
                        col - self.center[1], q)
        return M_pq

    def _calculate_N_pq(self, p: int, q: int) -> float:
        return self._calculate_M_pq(p, q) / pow(self.pixels_count(),
                                                ((p + q) / 2) + 1)

    def _calculate_M1_normalized(self) -> float:
        return self._calculate_N_pq(2, 0) + self._calculate_N_pq(0, 2)

    def _calculate_M7_normalized(self) -> float:
        N_20 = self._calculate_N_pq(2, 0)
        N_02 = self._calculate_N_pq(0, 2)
        N_11 = self._calculate_N_pq(1, 1)

        return N_20 * N_02 - pow(N_11, 2)

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


def segmentation(image: np.ndarray) -> List[List[Segment]]:
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
                            if USE_CLOSE_OP:
                                segment.update_image(
                                    _close_operator(segment.local_image, 2))

                            mask_objects_list.append(segment)

        # for i, object in enumerate(mask_objects_list):
        #     object.local_image.dtype = 'uint8'
        #     cv.imshow(str(i), object.local_image * 255)
        # cv.waitKey(0)

        objects_list.append(copy.deepcopy(mask_objects_list))
        mask_objects_list.clear()
        visited = np.zeros((image_height, image_width), dtype=bool)

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


def _close_operator(image: np.ndarray, n: int) -> np.ndarray:
    output = ranking_filter(image, (3, 3), np.any)
    for _ in range(n - 1):
        # dilation
        output = ranking_filter(output, (3, 3), np.any)
    for _ in range(n):
        # erosion
        output = ranking_filter(output, (3, 3), np.all)

    return output


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
