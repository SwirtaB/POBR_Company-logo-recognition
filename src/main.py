import sys
import cv2 as cv
import numpy as np
import color_converters as cc
import resizers as res
import processing as proc
from config import BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS
import segmentation as seg
import identyfication as idef
from typing import List, Tuple


def _get_logo_bbox(image: np.ndarray, logos_list: List[List[seg.Segment]]):
    image_height, image_width, _ = image.shape

    for logo in logos_list:
        min_row, max_row, min_col, max_col = image_height, 0, image_width, 0
        for rect in logo:
            if rect.bbox[0][0] < min_row:
                min_row = rect.bbox[0][0]
            if rect.bbox[1][0] > max_row:
                max_row = rect.bbox[1][0]
            if rect.bbox[0][1] < min_col:
                min_col = rect.bbox[0][1]
            if rect.bbox[1][1] > max_col:
                max_col = rect.bbox[1][1]

        cv.rectangle(image, (min_col, min_row), (max_col, max_row),
                     color=BOUNDING_BOX_COLOR,
                     thickness=BOUNDING_BOX_THICKNESS + 2)


def main() -> int:
    path = "./Data/ms_multi_logo.jpg"
    image = cv.imread(path)

    if image is None:
        sys.exit(f"Could not read image under path: {path}")

    resized = res.resize(image, np.uint(image.shape[1] / 8),
                         np.uint(image.shape[0] / 8),
                         res.bilinear_interpolation)
    equlized_hsv = proc.equalize_histogram(cc.BGR_to_HSV(resized))
    blured = proc.applay_convolution(cc.HSV_to_BGR(equlized_hsv),
                                     proc.BLUR_KERNEL_CLASSIC)
    # blured = proc.applay_convolution(resized, proc.BLUR_KERNEL_CLASSIC)
    segments = seg.segmentation(cc.BGR_to_HSV(blured))
    segments = idef.filter_segments(segments)

    for segment_list in segments:
        for segment in segment_list:
            p1 = (segment.bbox[0][1], segment.bbox[0][0])
            p2 = (segment.bbox[1][1], segment.bbox[1][0])
            cv.rectangle(resized,
                         p1,
                         p2,
                         color=BOUNDING_BOX_COLOR,
                         thickness=BOUNDING_BOX_THICKNESS)

    logos = idef.try_recognize_logo(segments)
    _get_logo_bbox(resized, logos)
    print("Break")
    cv.imshow("resized", resized)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
