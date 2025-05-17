import numpy as np
from scipy.signal import convolve2d

# Input dimensions: (3, 3, 3)
X = np.array([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]],

    [[2, 1, 2],
     [1, 2, 1],
     [2, 1, 2]]
])

# Kernels: (2 filters, 3 channels, 2x2)
K1 = np.array([
    [[1, 0], [0, 1]],  # K1[0]
    [[1, 1], [1, 1]],  # K1[1]
    [[0, 1], [1, 0]]   # K1[2]
])

K2 = np.array([
    [[0, 1], [1, 0]],  # K2[0]
    [[1, 0], [0, 1]],  # K2[1]
    [[0, 1], [1, 0]]   # K2[2]
])

# Output gradients: (2 filters, 2x2)
dY = np.array([
    [[1, 2],
     [3, 4]],

    [[5, 6],
     [7, 8]]
])

# Prepare flipped kernels (rotate 180°)
def flip_kernel(k):
    return np.rot90(k, 2)

K1_flipped = np.array([flip_kernel(k) for k in K1])
K2_flipped = np.array([flip_kernel(k) for k in K2])

# Gradient of input X
dX = np.zeros_like(X)

# Sum full convolutions of dY with flipped kernels per input channel
for c in range(3):  # input channels
    dX[c] += convolve2d(dY[0], K1_flipped[c], mode='full')
    dX[c] += convolve2d(dY[1], K2_flipped[c], mode='full')

# Print result
np.set_printoptions(suppress=True)
print(dX)

