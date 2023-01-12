import cv2 as cv
import sys
import color_converters as cc
import resizers as res
import processing as proc


def main() -> int:
    path = "./Data/ms_building_2.jpg"
    image = cv.imread(path)

    if image is None:
        sys.exit(f"Could not read image under path: {path}")

    resized = res.resize(image, 360, 640, res.bilinear_interpolation)
    blured = proc.applay_convolution(resized, proc.BLUR_KERNEL)
    cv.imshow("resized", resized)
    cv.imshow("image", blured)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
