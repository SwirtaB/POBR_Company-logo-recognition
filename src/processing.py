import numpy as np

BLUR_KERNEL = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9]])


def applay_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_width, kernel_height = kernel.shape
    shift_width = int(np.floor(kernel_width / 2))
    shift_height = int(np.floor(kernel_height / 2))
    image_height, image_width, channels = image.shape
    output = np.zeros(image.shape, dtype=np.uint8)

    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            for ch in range(channels):
                output[row][col][ch] = np.sum(np.multiply(
                    image[:, :, ch][row - shift_width:row + shift_width + 1,
                                    col - shift_height:col + shift_height + 1],
                    kernel),
                                              dtype=np.uint8)
    return output