import cv2 as cv
import sys
import color_converters as cc
import resizers as res
import processing as proc
import config
import segmentation as seg
import numpy as np


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
    cv.imshow("resized", resized)
    cv.imshow("fixed", blured)
    segments = seg.segmentation(cc.BGR_to_HSV(blured))
    print("Break")

    # resized = res.resize(image, 360, 640, res.bilinear_interpolation)
    # blured = proc.applay_convolution(resized, proc.BLUR_KERNEL_CLASSIC)
    # cv.imshow("resized", resized)
    # cv.imshow("image", blured)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
