import cv2 as cv
import sys


def main() -> int:
    path = "./Data/ms_logo_1.jpg"
    image = cv.imread(path)

    if image is None:
        sys.exit(f"Could not read image under path: {path}")

    cv.imshow("Loaded image", image)
    cv.waitKey(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())
