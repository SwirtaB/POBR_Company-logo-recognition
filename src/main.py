import cv2 as cv
import sys
import color_converters as cc
import resizers


def main() -> int:
    path = "./Data/ms_building_2.jpg"
    image = cv.imread(path)

    if image is None:
        sys.exit(f"Could not read image under path: {path}")

    # cv.imshow("HSV mine", cc.BGR_to_HSV(image))
    # cv.imshow("HSV lib", cv.cvtColor(image, cv.COLOR_BGR2HSV))
    # cv.waitKey(0)
    nn_resized = resizers.resize(image, 720, 1280, resizers.nn_interpolation)
    cv.imshow("nn resized", nn_resized)

    bl_resized = resizers.resize(image, 720, 1280,
                                 resizers.bilinear_interpolation)
    cv.imshow("bl esized", bl_resized)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
