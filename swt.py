from math import floor
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.draw import line as skline
from skimage.feature import canny
from skimage.filters import scharr_h, scharr_v
from skimage.io import imread
from swt_src import swt as swt2

Image = np.ndarray
SubImage = np.ndarray
GradientImg = np.ndarray
Gradients = Tuple[GradientImg, GradientImg]
Dimensions = Tuple[int, int]
Position = Tuple[int, int]
Gradient = Tuple[float, float]
Ray = Iterator[Tuple[int, int]]


def derivatives(img: Image, grad_dir: int) -> Tuple[Image, Gradients, Image]:
    # edge_map and x, y gradients and angles
    gradients = (scharr_h(img), scharr_v(img))
    angles = np.arctan2(gradients[1] * grad_dir, gradients[0] * grad_dir)
    return canny(img), gradients, angles


def project_ray(diag: float, start: Position, angles: Image) -> Ray:
    # Calculate the end point of the line
    end_x = floor(start[0] + diag * np.cos(angles[start]))
    end_y = floor(start[1] + diag * np.sin(angles[start]))
    skline(start[0], start[1], end_x, end_y)

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


def plot_new_line(img: Image, start: Position, end: Position):
    # calculate the distance between the edge point and the endpoint
    dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    line = skline(start[0], start[1], end[0], end[1])

    # Update pixels that belong to the new line with the minimum distance
    img[line] = np.minimum(img[line], dist)

    return line


def set_ray_above_median(ray: Ray, image: Image):
    median_value = np.median(image[ray])

    # set all pixels above the median to the median value
    image[ray][image[ray] >= median_value] = median_value


def angle_between(start, end, gradients):
    v1 = (gradients[0][start], gradients[1][start])
    v2 = (gradients[0][end], gradients[1][end])

    dotmag = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.abs(np.arccos(np.clip(dotmag, -1, 1)) - np.pi)
    return angle


def swt(img: Image, edge_map: Image, gradients: Gradients, angles: Image) -> Image:
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

        # project a line from the edge point in the direction of the gradient
        ray = project_ray(diaglen, start, angles)

        # track if endpoint is found
        end = traverse_line(ray, dims, edge_map)

        if end is None:
            continue

        if angle_between(start, end, gradients) > np.pi / 6:
            continue

        # plot the new line
        new_ray = plot_new_line(projection, start, end)
        rays.append(new_ray)

    # set all the upper bounds back to 0
    projection[projection == np.Infinity] = 0

    for ray in rays:
        set_ray_above_median(ray, projection)

    return projection


def store_equivalent_labels(eq_labels: dict, label1, label2):
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
    eq_labels = {}

    def neighbors(i, j):
        yield i - 1, j
        yield i, j - 1
        yield i - 1, j - 1

    for i, j in np.argwhere(swt != 0):

        def keep(x1, y1):
            max_ratio = 2
            ratio = np.abs(swt[x1, y1] / swt[i, j])
            ratio = ratio > 1 / max_ratio and ratio < max_ratio
            return (labels[x1, y1] != 0) and ratio

        neighs = [labels[x, y] for x, y in neighbors(i, j) if keep(x, y)]
        neighs = sorted(list(set(neighs)))

        if not neighs:
            labels[i, j] = label = label + 1
            continue

        labels[i, j] = neighs[0]

        for n in neighs[1:]:
            store_equivalent_labels(eq_labels, neighs[0], n)

    for i, j in np.argwhere(labels != 0):
        root = labels[i, j]
        while eq_labels.get(root, None) is not None:
            root = eq_labels[root]
        labels[i, j] = root

    reduced_labels = sorted(list(set(labels.ravel())))
    for i, label in enumerate(reduced_labels):
        labels[labels == label] = i

    return labels


def connected(swt: Image):
    def is_connected(p1, p2):
        return p1 != 0 and p2 != 0 and abs(p1 / p2) < 3 and abs(p2 / p1) < 3

    # Label connected components using scikit-image
    labeled_image = measure.label(swt, connectivity=2, background=0)

    # Iterate over all labeled regions and check their connectivity
    for region in measure.regionprops(labeled_image, intensity_image=swt):
        region_intensity = region.mean_intensity
        for coord in region.coords:
            pixel_intensity = swt[coord[0], coord[1]]
            if not is_connected(region_intensity, pixel_intensity):
                labeled_image[coord[0], coord[1]] = 0

    return labeled_image


def compare():
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    ax = axes.ravel()

    # img = imread("../Images/test/text-0.png", as_gray=True)
    img = imread("../Images/test/text.jpg", as_gray=True)
    # img = imread("../Images/test/swt-test.png", as_gray=True)

    edges, gradients, angles = derivatives(img, -1)
    swt_img = swt(img, edges, gradients, angles)

    edges2, angles2, swt_img2 = swt2(img)

    ax[0].set_title("SK Edge Map")
    ax[0].imshow(edges, cmap="gray")

    ax[1].set_title("Gradient Map")
    ax[1].imshow(angles * 180 / np.pi, cmap="gray")

    ax[2].set_title("My SWT")
    ax[2].imshow(swt_img, cmap="gray")

    ax[3].set_title("CV Edge Map")
    ax[3].imshow(edges2, cmap="gray")

    ax[4].set_title("Gradient Map")
    ax[4].imshow(angles2 * 180 / np.pi, cmap="gray")

    ax[5].set_title("OG SWT")
    ax[5].imshow(swt_img2, cmap="gray")

    plt.show()


def compare2():
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.ravel()
    img = imread("../Images/test/swt-test.png", as_gray=True)
    # img = imread("../Images/test/text.jpg", as_gray=True)

    edges, gradients, angles = derivatives(img, -1)
    swt_img = swt(img, edges, gradients, angles)
    edges, gradients, angles = derivatives(img, 1)
    swt_img2 = swt(img, edges, gradients, angles)

    ax[0].set_title("SWT")
    ax[0].imshow(swt_img, cmap="gray")

    ax[1].set_title("SWT")
    ax[1].imshow(swt_img2, cmap="gray")

    plt.show()


def test():
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    img = imread("../Images/test/text-0.png", as_gray=True)
    # img = imread("../Images/test/text.jpg", as_gray=True)

    edges, gradients, angles = derivatives(img, -1)
    swt_img = swt(img, edges, gradients, angles)

    axes.set_title("SWT")
    axes.imshow(swt_img, cmap="gray")

    plt.show()


if __name__ == "__main__":
    compare()
    # compare2()
    # test()
