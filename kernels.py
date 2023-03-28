import math
from collections import Counter
from math import pi
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from skimage.draw import line

dim = 101
h_dim = dim // 2
domain = 145

kernels3 = {
    "basic3": {
        "x": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        "y": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    },
    "sobel3": {
        "x": [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        "y": [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    },
    "scharr3": {
        "x": [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
        "y": [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
    },
    "optimum3": {
        "x": [[-0.112737, 0, 0.112737], [-0.274526, 0, 0.274526], [-0.112737, 0, 0.112737]],
        "y": [[-0.112737, -0.274526, -0.112737], [0, 0, 0], [0.112737, 0.274526, 0.112737]]
    },
}

angles3 = {
    "angle": [i for i in range(domain)],
    "basic3": [],
    "sobel3": [],
    "scharr3": [],
    "optimum3": [],
}

kernels5 = {
    "sobel5": {

        "x": [[2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2],
              [4, 2, 0, -2, -4],
              [2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2]],

        "y": [[2, 2, 4, 2, 2],
              [1, 1, 2, 1, 1],
              [0, 0, 0, 0, 0],
              [-1, -1, -2, -1, -1],
              [-2, -2, -4, -2, -2]],
    },

    "previtt5": {
        "x": [[2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2],
              [2, 1, 0, -1, -2]],

        "y": [[2, 2, 2, 2, 2],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [-1, -1, -1, -1, -1],
              [-2, -2, -2, -2, -2]],
    },

    "kirsch5": {
        "x": [[9, 9, 9, 9, 9],
              [9, 5, 5, 5, 9],
              [-7, -3, 0, -3, -7],
              [-7, -3, -3, -3, -7],
              [-7, -7, -7, -7, -7]],

        "y": [[9, 9, -7, -7, -7],
              [9, 5, -3, -3, -7],
              [9, 5, 0, -3, -7],
              [9, 5, -3, -3, -7],
              [9, 9, -7, -7, -7]]
    },

    "optimum5": {
        "x": [[-0.003776, -0.010199, 0, 0.010199, 0.003776],
              [-0.026786, -0.070844, 0, 0.070844, 0.026786],
              [-0.046548, -0.122572, 0, 0.122572, 0.046548],
              [-0.026786, -0.070844, 0, 0.070844, 0.026786],
              [-0.003776, -0.010199, 0, 0.010199, 0.003776]],

        "y": [[-0.003776, -0.026786, -0.046548, -0.026786, -0.003776],
              [-0.010199, -0.070844, -0.122572, -0.070844, -0.010199],
              [0, 0, 0, 0, 0],
              [0.010199, 0.070844, 0.122572, 0.070844, 0.010199],
              [0.003776, 0.026786, 0.046548, 0.026786, 0.003776]],
    },
}

angles5 = {
    "angle": [i for i in range(domain)],
    "sobel5": [],
    "previtt5": [],
    "kirsch5": [],
    "optimum5": [],
}


def loop_kernels(kernels, img, output):
    for name, kernel in kernels.items():
        x = convolve2d(img, kernel["x"], mode="same")
        y = convolve2d(img, kernel["y"], mode="same")
        angles = (np.arctan2(x, y) * 180 / pi) + 90
        final = Counter(angles.ravel()).most_common(2)[-1][0]
        output[name].append(final % 180)


def draw_line(angle):
    angle *= pi / 180
    x = math.floor(h_dim * math.sin(angle))
    y = math.floor(h_dim * math.cos(angle))
    rr, cc = line(h_dim - x, h_dim - y, h_dim + x, h_dim + y)
    img = np.zeros((dim, dim))
    img[rr, cc] = 1
    return img


def save_csv(name, kernel):
    df = pd.DataFrame(kernel)
    df.to_csv(f"{name}.csv", index=False)


if __name__ == "__main__":
    for angle in range(domain):
        img = draw_line(angle)
        loop_kernels(kernels3, img, angles3)
        # loop_kernels(kernels5, img, angles5)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for name, kernel in angles3.items():
        mse = np.mean((np.array(kernel) - np.array(angles3["angle"])) ** 2)
        # mse = 1
        ax.plot(kernel, label=f"{name}, mse={math.sqrt(mse):.2f}")
    ax.legend()
    plt.show()
