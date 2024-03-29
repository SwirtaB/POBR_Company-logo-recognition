import sys
import cv2 as cv
import numpy as np
import argparse
import pathlib
from typing import List

import color_converters as cc
import resizers as res
import processing as proc
import segmentation as seg
import identyfication as idef
from config import BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS, GAUSS_SIGMA, GAUSS_KERNEL_SIZE, VERBOSE


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
                     thickness=BOUNDING_BOX_THICKNESS + 1)


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="POBR - Microsoft Logo Recognition Pipeline")

    group = parser.add_argument_group()
    group.add_argument("-f",
                       "--file",
                       help="path to image file",
                       type=pathlib.Path)
    group.add_argument("-s",
                       "--scale",
                       help="scale down factor for image",
                       type=int)
    group.add_argument("-b",
                       "--blure",
                       help="blure method to use: classic(default), gauss",
                       type=str)

    return parser.parse_args()


def main() -> int:
    args = _get_arguments()
    try:
        if args.file is None:
            print("Missing path to file")
            return 0
        else:
            image = cv.imread(str(args.file))
            if image is None:
                sys.exit(f"Could not read image under path: {str(args.file)}")

            scale = 4
            if args.scale is not None:
                scale = args.scale
            resized = res.resize(image, np.uint(image.shape[1] / scale),
                                 np.uint(image.shape[0] / scale),
                                 res.bilinear_interpolation)
            equlized_hsv = proc.equalize_histogram(cc.BGR_to_HSV(resized))

            blure_kernel = proc.BLUR_KERNEL_CLASSIC
            if args.blure is not None and args.blure == "gauss":
                blure_kernel = proc.generate_gausse_kernel(
                    GAUSS_SIGMA, GAUSS_KERNEL_SIZE)

            blured = proc.applay_convolution(cc.HSV_to_BGR(equlized_hsv),
                                             blure_kernel)
            segments = seg.segmentation(cc.BGR_to_HSV(blured))
            segments = idef.filter_segments(segments)

            if VERBOSE:
                for segment_list in segments:
                    for segment in segment_list:
                        p1 = (segment.bbox[0][1], segment.bbox[0][0])
                        p2 = (segment.bbox[1][1], segment.bbox[1][0])
                        cv.rectangle(resized,
                                     p1,
                                     p2,
                                     color=BOUNDING_BOX_COLOR,
                                     thickness=BOUNDING_BOX_THICKNESS)

                cv.imshow("blured", blured)

            logos = idef.try_recognize_logo(segments)
            _get_logo_bbox(resized, logos)
            cv.imshow("resized", resized)
            cv.waitKey(0)
            return 0

    except Exception as error:
        print(error)
        return -1


if __name__ == '__main__':
    sys.exit(main())
