import cv2
import numpy as np
import numba as nb
from timeit import timeit

@nb.njit(cache=True, fastmath=True)
def norm_images(images):
    image_blocks = np.empty((len(images), 200, 200))
    for i in range(len(images)):
        imin = images[i].min()
        imax = images[i].max()
        image_blocks[i, :, :] = (images[i] - imin)/(imax-imin)
    return image_blocks

def load_image_blocks():
    images = [cv2.imread(f'data/{i}.jpg', 0) for i in range(100)]
    out = norm_images(images)
    return out

def load_normally():
    out = np.empty((100, 200, 200), np.float32)
    for i in range(100):
        image = cv2.imread(f'data/{i}.jpg', 0)
        out[i, :, :] = (image-image.min())/(image.max()-image.min())
    return out
if __name__ == '__main__':
    load_image_blocks()
    n = 5
    a = timeit(lambda : load_image_blocks(), number=n)
    b = timeit(lambda : load_normally(), number=n)
    print(a, b)
