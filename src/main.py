import sys
import cv2 as cv
import numpy as np
import color_converters as cc
import resizers as res
import processing as proc
from config import BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS
import segmentation as seg


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
    for segment_list in segments:
        for segment in segment_list:
            p1 = (segment.bbox[0][1], segment.bbox[0][0])
            p2 = (segment.bbox[1][1], segment.bbox[1][0])
            cv.rectangle(resized,
                         p1,
                         p2,
                         color=BOUNDING_BOX_COLOR,
                         thickness=BOUNDING_BOX_THICKNESS)
    print("Break")
    cv.imshow("resized", resized)
    cv.imshow("fixed", blured)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
