import math
import os
from math import pi
from typing import List, NamedTuple, Optional, Tuple, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import line
from skimage.feature import canny
from skimage.filters import scharr_h, scharr_v
from skimage.io import imread

Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple("Position", [("x", int), ("y", int)])
Stroke = NamedTuple("Stroke", [("x", int), ("y", int), ("width", float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar("ImageOrValue", float, Image)
Gradients = NamedTuple("Gradients", [("x", GradientImage), ("y", GradientImage)])


def gamma(x: ImageOrValue, coeff: float = 2.2) -> ImageOrValue:
    """
    Applies a gamma transformation to the input.

    :param x: The value to transform.
    :param coeff: The gamma coefficient to use.
    :return: The transformed value.
    """
    return x ** (1.0 / coeff)


def gleam(im: Image, gamma_coeff: float = 2.2) -> Image:
    """
    Implements Gleam grayscale conversion from
    Kanan & Cottrell 2012: Color-to-Grayscale: Does the Method Matter in Image Recognition?
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740

    :param im: The image to convert.
    :param gamma_coeff: The gamma coefficient to use.
    :return: The grayscale converted image.
    """
    im = gamma(im, gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)


def open_grayscale(path: str) -> Image:
    """
    Opens an image and converts it to grayscale.

    :param path: The image to open.
    :return: The grayscale image.
    """
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = im.astype(np.float32) / 255.0
    return gleam(im)


def get_edges(im: Image, lo: float = 175, hi: float = 220, window: int = 3) -> Image:
    """
    Detects edges in the image by applying a Canny edge detector.

    :param im: The image.
    :param lo: The lower threshold.
    :param hi: The higher threshold.
    :param window: The window (aperture) size.
    :return: The edges.
    """
    # OpenCV's Canny detector requires 8-bit inputs.
    im = (im * 255.0).astype(np.uint8)
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.0


def get_gradients(im: Image) -> Gradients:
    """
    Obtains the image gradients by means of a 3x3 Scharr filter.

    :param im: The image to process.
    :return: The image gradients.
    """
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def get_gradient_directions(g: Gradients) -> Image:
    """
    Obtains the gradient directions.

    :param g: The gradients.
    :return: An image of the gradient directions.
    """
    return np.arctan2(g.y, g.x)


def apply_swt(
    im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool = True
) -> Image:
    """
    Applies the Stroke Width Transformation to the image.

    :param im: The image
    :param edges: The edges of the image.
    :param gradients: The gradients of the image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    :return: The transformed image.
    """
    # Prepare the output map.
    swt = np.ones(im.shape) * np.Infinity

    # For each pixel, let's obtain the normal direction of its gradient.

    norms = np.sqrt(gradients.x**2 + gradients.y**2)
    norms[norms == 0] = 1
    inv_norms = 1.0 / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)

    # We keep track of all the rays found in the image.
    rays = []

    rr, cc = np.where(edges == 1)
    for y, x in zip(rr, cc):
        ray = swt_process_pixel(
            Position(x=x, y=y),
            edges,
            directions,
            out=swt,
            dark_on_bright=dark_on_bright,
        )
        if ray:
            rays.append(ray)

    # Multiple rays may cross the same pixel and each pixel has the smallest
    # stroke width of those.
    # A problem are corners like the edge of an L. Here, two rays will be found,
    # both of which are significantly longer than the actual width of each
    # individual stroke. To mitigate, we will visit each pixel on each ray and
    # take the median stroke length over all pixels on the ray.
    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            # swt[p.y, p.x] = min(median, swt[p.y, p.x])
            swt[p.y, p.x] = 1

    swt[swt == np.Infinity] = 0
    return swt


def project_line(angle_img, x, y):
    def return_line(x, y, x2, y2):
        rr, cc = line(x, y, x2, y2)
        return zip(rr, cc)

    theta = angle_img[x][y] + pi / 2

    if theta == pi / 2:
        return return_line(x, y, x, len(angle_img[0]) - 1)
    elif theta == -pi / 2:
        return return_line(x, y, x, 0)

    m = math.tan(theta)
    c = y - m * x

    newX = 0 if theta < 0 else len(angle_img) - 1
    newY = m * newX + c

    if newY < 0:
        newY = 0
        newX = (newY - c) / m
    elif newY > len(angle_img[0]) - 1:
        newY = len(angle_img[0]) - 1
        newX = (newY - c) / m

    return return_line(x, y, math.floor(newX), math.floor(newY))


def swt_process_pixel(
    pos: Position,
    edges: Image,
    directions: Gradients,
    out: Image,
    dark_on_bright: bool = True,
) -> Optional[Ray]:
    """
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    """
    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    # The direction in which we travel the gradient depends on the type of text
    # we want to find. For dark text on light background, follow the opposite
    # direction (into the dark are); for light text on dark background, follow
    # the gradient as is.
    gradient_direction = -1 if dark_on_bright else 1

    # Starting from the current pixel we will shoot a ray into the direction
    # of the pixel's gradient and keep track of all pixels in that direction
    # that still lie on an edge.
    ray = [pos]

    # Obtain the direction to step into
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edge before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y))

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line.
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        # The point is either on the line or the end of it, so we register it.
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and
        # need to continue scanning.
        if edges[cur_y, cur_x] < 0.5:  # TODO: Test for image boundaries here
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6.
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y
        if dot_product >= -0.866:
            return None
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt(
            (cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y)
        )
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])
        return ray

    # noinspection PyUnreachableCode
    assert False, "This code cannot be reached."


def connected_components(
    swt: Image, threshold: float = 3.0
) -> Tuple[Image, List[Component]]:
    """
    Applies Connected Components labeling to the transformed image using a flood-fill algorithm.

    :param swt: The Stroke Width transformed image.
    :param threshold: The Stroke Width ratio below which two strokes are considered the same.
    :return: The map of labels.
    """
    height, width = swt.shape[0:2]
    labels = np.zeros_like(swt, dtype=np.uint32)
    next_label = 0
    components = []  # List[Component]
    for y in range(height):
        for x in range(width):
            # current pixel swt value
            stroke_width = swt[y, x]

            # disregard pixels with no stroke value as well as already labeled pixels
            if (stroke_width <= 0) or (labels[y, x] > 0):
                continue

            next_label += 1
            neighbor_labels = [Stroke(x=x, y=y, width=stroke_width)]
            component = []
            while len(neighbor_labels) > 0:
                neighbor = neighbor_labels.pop()
                npos, stroke_width = (
                    Position(x=neighbor.x, y=neighbor.y),
                    neighbor.width,
                )
                if not ((0 <= npos.x < width) and (0 <= npos.y < height)):
                    continue
                # If the current pixel was already labeled, skip it.
                n_label = labels[npos.y, npos.x]
                if n_label > 0:
                    continue
                # We associate pixels based on their stroke width. If there is no stroke, skip the pixel.
                n_stroke_width = swt[npos.y, npos.x]
                if n_stroke_width <= 0:
                    continue
                # We consider this point only if it is within the acceptable threshold and in the initial test
                # (i.e. when visiting a new stroke), the ratio is 1.
                # If we succeed, we can label this pixel as belonging to the same group. This allows for
                # varying stroke widths due to e.g. perspective distortion or elaborate fonts.
                if (stroke_width / n_stroke_width >= threshold) or (
                    n_stroke_width / stroke_width >= threshold
                ):
                    continue
                labels[npos.y, npos.x] = next_label
                component.append(npos)
                # From here, we're going to expand the new neighbors.
                neighbors = {
                    Stroke(x=npos.x - 1, y=npos.y - 1, width=n_stroke_width),
                    Stroke(x=npos.x, y=npos.y - 1, width=n_stroke_width),
                    Stroke(x=npos.x + 1, y=npos.y - 1, width=n_stroke_width),
                    Stroke(x=npos.x - 1, y=npos.y, width=n_stroke_width),
                    Stroke(x=npos.x + 1, y=npos.y, width=n_stroke_width),
                    Stroke(x=npos.x - 1, y=npos.y + 1, width=n_stroke_width),
                    Stroke(x=npos.x, y=npos.y + 1, width=n_stroke_width),
                    Stroke(x=npos.x + 1, y=npos.y + 1, width=n_stroke_width),
                }
                neighbor_labels.extend(neighbors)
            if len(component) > 0:
                components.append(component)
    return labels, components


def swt(img):
    gradients = get_gradients(img)
    edges = get_edges(img)
    theta = get_gradient_directions(gradients)
    return edges, theta, apply_swt(img, edges, gradients, True)


def main():
    im = open_grayscale("C:/Personal/Coding/Meesters/Stroke Width/scripts/SWT/text.jpg")
    Canny = imread("../Images/test/text.jpg", as_gray=True)
    edges = get_edges(im)
    gradients = get_gradients(im)
    Canny = canny(Canny)
    # theta = get_gradient_directions(gradients)
    # theta = np.abs(theta)
    # swt_cv = apply_swt(im, edges, gradients, True)
    swt_sk = apply_swt(im, Canny, gradients, True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.ravel()

    # ax[0].imshow(edges, cmap='gray')
    # ax[1].imshow(Canny, cmap='gray')
    # ax[2].imshow(swt_cv, cmap='gray')
    # ax[3].imshow(swt_sk, cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()
