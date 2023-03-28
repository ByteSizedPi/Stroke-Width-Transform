from math import floor
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line as skline
from skimage.feature import canny
from skimage.filters import scharr_h, scharr_v
from skimage.io import imread
from skimage.measure import label

Image = np.ndarray
SubImage = np.ndarray
GradientImg = np.ndarray
Gradients = Tuple[GradientImg, GradientImg]
Dimensions = Tuple[int, int]
Position = Tuple[int, int]
Gradient = Tuple[float, float]
Ray = Iterator[Tuple[int, int]]


def project_ray(diag: float, start: Position, gradient: Gradient, grad_dir: int) -> Ray:
    x_grad, y_grad = gradient
    # Calculate the angle of the gradient
    angle = np.arctan2(y_grad * grad_dir, x_grad * grad_dir)

    # Calculate the end point of the line
    end_x = floor(start[0] + diag * np.cos(angle))
    end_y = floor(start[1] + diag * np.sin(angle))

    return zip(*skline(start[0], start[1], end_x, end_y))


def traverse_line(line: Ray, dims: Dimensions, edge_map: Image) -> Position:
    # track if endpoint is found
    end = None

    # skip the current position
    next(line)

    for r, c in line:
        # out of bounds check
        if (r < 0) or (r >= dims[0]) or (c < 0) or (c >= dims[1]):
            break

        # other edge found
        if edge_map[r, c] == 1:
            end = (r, c)
            break

    return end


def angle_diff(gradient, gradients: Gradients, end: Position) -> bool:
    # calculate the angle at both points
    startAngle = np.arctan2(gradient[1], gradient[0])
    endAngle = np.arctan2(gradients[1][end], gradients[0][end])

    # calculate the difference between the angles
    angle = np.abs(startAngle - (endAngle + np.pi) % np.pi)
    return angle > np.pi / 6


def plot_new_line(img: Image, start: Position, end: Position):
    # calculate the distance between the edge point and the endpoint
    dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    line = skline(start[0], start[1], end[0], end[1])

    # a pixel is equal to the shortest stroke it is part of
    for idx in zip(*line):
        if (img[idx] > dist) or (img[idx] == np.Infinity):
            img[idx] = dist

    return line


def set_ray_above_median(ray: Ray, image: Image):
    # get pixel values for each pixel in the ray
    pixel_values = [image[pixel] for pixel in zip(*ray)]

    # compute the median pixel value
    median_value = np.median(pixel_values)

    # set all pixels above the median to the median value
    for pixel in zip(*ray):
        if image[pixel] >= median_value:
            image[pixel] = median_value


def swt(img: Image, edge_map: Image, gradients: Gradients, grad_dir: int) -> Image:
    # Output image
    projection = np.ones(img.shape) * np.Infinity

    # collection of rays
    rays = []

    # image dimensions
    dims = (img.shape[0], img.shape[1])

    # diagonal of image for initial line projection length
    diaglen = np.sqrt(dims[0] ** 2 + dims[1] ** 2)

    # loop the edges in the edge map
    for edge_x, edge_y in np.argwhere(edge_map == 1):
        start = edge_x, edge_y

        # x and y gradient at current point
        gradient = (gradients[0][start], gradients[1][start])

        # project a line from the edge point in the direction of the gradient
        ray = project_ray(diaglen, start, gradient, grad_dir)

        # track if endpoint is found
        end = traverse_line(ray, dims, edge_map)

        if end is None:
            continue

        # a difference of more than pi/6 is not a valid edge
        if angle_diff(gradient, gradients, end):
            continue

        # plot the new line
        new_ray = plot_new_line(projection, start, end)
        rays.append(new_ray)

    # set all the upper bounds back to 0
    projection[projection == np.Infinity] = 0

    for ray in rays:
        set_ray_above_median(ray, projection)

    return projection


def store_equivalent_labels(eq_labels, label1, label2):
    """
    Store the equivalence relationship between two labels in a dictionary.

    Args:
        eq_labels (dict): A dictionary of equivalent labels.
        label1 (int): The first label to be stored.
        label2 (int): The second label to be stored.
    """
    # Find the root labels for the two labels
    root1 = label1
    while eq_labels.get(root1, None) is not None:
        root1 = eq_labels[root1]
    root2 = label2
    while eq_labels.get(root2, None) is not None:
        root2 = eq_labels[root2]

    # Store the equivalence relationship
    if root1 != root2:
        eq_labels[root2] = root1


def connected_components(swt: Image):
    label = 0
    labels = np.zeros(swt.shape, dtype=np.int32)

    def neighbors(i, j):
        yield i - 1, j
        yield i, j - 1
        yield i - 1, j - 1

    for i, j in np.argwhere(swt != 0):
        neighs = [labels[x, y] for x, y in neighbors(i, j) if labels[x, y] != 0]
        neighs = list(set(neighs))

        if not neighs:
            labels[i, j] = label = label + 1
            continue

        labels[i, j] = min(neighs)

    return labels


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.ravel()

    img = imread("../Images/test/text-0.png", as_gray=True)
    # declare a new image type

    # img = imread("../Images/test/text.jpg", as_gray=True)

    canny_img = canny(img)
    gradients = (scharr_h(img), scharr_v(img))
    swt_img = swt(img, canny_img, gradients, -1)
    # con_img = label(
    #     swt_img,
    #     # img
    #     connectivity=2,
    #     # return_num=False,
    # )
    con_img = connected_components(swt_img)

    ax[0].set_title("SWT")
    ax[0].imshow(swt_img, cmap="gray")

    ax[1].set_title("Connected Components")
    ax[1].imshow(con_img, cmap="gray")
    plt.show()
