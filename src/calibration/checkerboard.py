import cv2
import numpy as np

board = np.zeros((7*25*4, 10*25*4), np.uint8)

for r in range(7):
    for c in range(10):
        if (r+c) % 2 == 0:
            board[r*100:(r+1)*100, c*100:(c+1)*100] = 255

cv2.imwrite('src/calibration/checkerboard.png', board)
print('Saved checkerboard.png - print at exactly 100px = 25mm')
