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
            h = 0  # suppress possible unbound error
            if r == g == b:
                h = 0
            elif v == r:
                h = 60 * (((g - b) / diff) % 6)
            elif v == g:
                h = 60 * (b - r) / diff + 120
            elif v == b:
                h = 60 * (r - g) / diff + 240

            imageHSV[y][x] = np.array([h / 2, s * 255, v * 255],
                                      dtype=np.uint8)

    return imageHSV


def HSV_to_BGR(image: np.ndarray) -> np.ndarray:
    imageBGR = np.zeros(image.shape, dtype=np.uint8)

    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            h, s, v = pixel
            h, s, v = h * 2, s / 255, v / 255
            c = v * s
            t = c * (1 - abs(((h / 60) % 2) - 1))

            r, g, b = 0, 0, 0
            if h < 60:
                r, g, b = c, t, 0
            elif h < 120:
                r, g, b = t, c, 0
            elif h < 180:
                r, g, b = 0, c, t
            elif h < 240:
                r, g, b = 0, t, c
            elif h < 300:
                r, g, b = t, 0, c
            else:
                r, g, b = c, 0, t

            m = v - c
            imageBGR[y][x] = np.array([(b + m) * 255, (g + m) * 255,
                                       (r + m) * 255],
                                      dtype=np.uint8)

    return imageBGR


# print(
#     BGR_to_HSV(
#         np.array([[[64, 93, 180], [140, 162, 250], [33, 138, 250],
#                    [88, 45, 128]]])))
