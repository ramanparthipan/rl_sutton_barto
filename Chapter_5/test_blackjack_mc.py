import blackjack_mc
import numpy as np
import random

from skimage.draw import line

if __name__ == "__main__":
    img = np.zeros((10, 10), dtype=np.uint8)
    rr, cc = line(2, 2, 7, 7)
    img[rr, cc] = 1
    print(img)
    pass



