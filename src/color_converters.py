import numpy as np


def BGR_to_HSV(image: np.ndarray) -> np.ndarray:
    imageHSV = np.zeros(image.shape, dtype=np.uint8)

    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            b, g, r = pixel / 255
            cmin = min(b, g, r)

            #calc v
            v = max(b, g, r)
            diff = v - cmin

            #calc s
            if v != 0:
                s = diff / v
            else:
                s = 0

            #calc hue
            if r == g == b:
                h = 0
            elif v == r:
                h = 60 * (g - b) / diff
            elif v == g:
                h = 60 * (b - r) / diff + 120
            elif v == b:
                h = 60 * (r - g) / diff + 240

            imageHSV[y][x] = np.array([h / 2, s * 255, v * 255],
                                      dtype=np.uint8)

    return imageHSV