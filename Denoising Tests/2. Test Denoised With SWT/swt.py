from itertools import islice
from math import floor

import numpy as np
from skimage.draw import line as skline
from skimage.feature import canny
from skimage.filters import scharr_h, scharr_v


def SWT(img, grad_dir):
    def derivatives(img, grad_dir=1):
        # edge_map and x, y gradients and angles
        grads = (scharr_h(img), scharr_v(img))
        angles = np.arctan2(grads[1] * grad_dir, grads[0] * grad_dir)
        edges = canny(img)

        return edges, grads, angles

    def project_ray(diag, start, angles):
        # Calculate the end point of the line
        end_x = floor(start[0] + diag * np.cos(angles[start]))
        end_y = floor(start[1] + diag * np.sin(angles[start]))

        return zip(*skline(start[0], start[1], end_x, end_y))

    def traverse_line(line, dims, edge_map):
        # track if endpoint is found
        end = None

        height, width = dims
        # skip the current position
        line = islice(line, 1, None)

        for r, c in line:
            # out of bounds check
            if not (0 <= r < height and 0 <= c < width):
                break

            # other edge found
            if edge_map[r, c] == 1:
                end = (r, c)
                break

        return end

    def angle_between(start, end, gradients):
        # x, y gradients at the start and end points
        v1 = (gradients[0][start], gradients[1][start])
        v2 = (gradients[0][end] * -1, gradients[1][end] * -1)
        # np.negative(v2, out=v2)

        # determine angle by using the formula for the dot product
        dotmag = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # range of arccos is [0, pi]
        return np.arccos(dotmag)

    def plot_new_line(img, start, end):
        # calculate the distance between the edge point and the endpoint
        # distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        distance = np.hypot(end[0] - start[0], end[1] - start[1])
        line = skline(start[0], start[1], end[0], end[1])

        # Update pixels that belong to the new line with the minimum distance
        img[line] = np.minimum(img[line], distance)
        # np.minimum(img[line], distance, out=img[line])

        return line

    def swt(img, edge_map, gradients, angles):
        # Output image
        projection = np.ones(img.shape) * np.Infinity

        # collection of rays
        rays = []

        # image dimensions
        dims = (img.shape[0], img.shape[1])

        # diagonal of image for initial line projection length
        # diaglen = np.sqrt(dims[0] ** 2 + dims[1] ** 2)
        diaglen = np.linalg.norm(dims)

        # loop the edges in the edge map
        for edge_x, edge_y in np.argwhere(edge_map == 1):

            start = edge_x, edge_y

            # project a line from the edge point in the direction of the gradient
            ray = project_ray(diaglen, start, angles)

            # track if endpoint is found
            end = traverse_line(ray, dims, edge_map)

            if end is None:
                continue

            # lines that are not roughly opposite are discarded
            if angle_between(start, end, gradients) > np.pi / 4:
                continue

            # plot the new line
            new_ray = plot_new_line(projection, start, end)
            rays.append(new_ray)

        # set all the upper bounds back to 0
        projection[projection == np.Infinity] = 0

        medians = [np.median(projection[ray]) for ray in rays]

        for ray, median in zip(rays, medians):
            np.minimum(projection[ray], median, out=projection[ray])

        return rays, projection

    edge_map, grads, angles = derivatives(img, grad_dir)
    rays, swt_result = swt(img, edge_map, grads, angles)

    # Connected Components
    def store_equivalent_labels(eq_labels: dict, label1: int, label2: int):
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

    def connected_components(swt_img):
        label = 0
        labels = np.zeros(swt_img.shape, dtype=np.int32)
        eq_labels = {}

        def valid_neighbors(idx):
            # neighbors iterator for 8-connectivity
            def neighbors(idx):
                i, j = idx
                for x in range(i - 2, i + 3):
                    for y in range(j - 2, j + 3):
                        if (x, y) != (i, j):
                            yield x, y

            # return an array of indices of valid neighbors
            # for each possible neighbor of the edge pixel at idx
            # it is a possible neightbor if it is an edge pixel
            # we check if the label
            # return [labels[p] for p in neighbors(idx) if edges[p] and labels[p]]
            return [
                labels[p]
                for p in neighbors(idx)
                if p[0] < labels.shape[0]
                and p[1] < labels.shape[1]
                and swt_img[p] > 0
                and labels[p] != 0
                and ((swt_img[idx] / swt_img[p]) < 3)
                and ((swt_img[p] / swt_img[idx]) < 3)
            ]

        for edge_idx in np.argwhere(swt_img != 0):
            i, j = edge_idx

            # get the label values of the valid neighbors of the current point
            neighs = valid_neighbors((i, j))

            # if no neighbors, create a new label
            if not neighs:
                label += 1
                labels[i, j] = label
                continue

            # if one or more neighbor, assign the current point that label
            labels[i, j] = neighs[0]

            # store all labels as equivalent
            for n in neighs[1:]:
                store_equivalent_labels(eq_labels, neighs[0], n)

        # replace all labels with their root labels
        for i, j in np.argwhere(labels != 0):
            root = labels[i, j]
            while eq_labels.get(root, None) is not None:
                root = eq_labels[root]
            labels[i, j] = root

        unique = np.unique(labels)
        component_dict = {l: [] for l in unique if l != 0}

        for index, value in np.ndenumerate(labels):
            if value != 0:
                component_dict[value].append(index)

        return component_dict, labels

    component_dict, _ = connected_components(swt_result)

    # Bounding boxes

    def bounding_boxes(connected):

        boxes = []

        for _, indices in connected.items():
            xs = [i[0] for i in indices]
            ys = [i[1] for i in indices]

            xmin, xmax = np.min(xs), np.max(xs)
            ymin, ymax = np.min(ys), np.max(ys)
            width, height = xmax - xmin, ymax - ymin

            if width < 5 or height < 5:
                continue

            if (width / height) < 0.1 or (width / height) > 10:
                continue

            diameter = np.sqrt(width**2 + height**2)
            component = [swt_result[i] for i in indices]
            median = np.median(component)

            if diameter / median > 10:
                continue

            boxes.append((xmin, ymin, width, height))

        return boxes

    boxes = bounding_boxes(component_dict)

    def is_inside(box1, box2):
        x1min, y1min, width1, height1 = box1
        x2min, y2min, width1, height2 = box2

        return (
            x1min >= x2min
            and y1min >= y2min
            and x1min + width1 <= x2min + width1
            and y1min + height1 <= y2min + height2
        )

    box_counts = {i: 0 for i in range(len(boxes))}

    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i != j and is_inside(box2, box1):
                box_counts[i] += 1

    filtered_boxes = [box for i, box in enumerate(boxes) if box_counts[i] <= 2]
    return filtered_boxes
