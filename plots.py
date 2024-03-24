from math import ceil, sqrt

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)


def dims(i):
    c = ceil(sqrt(i))
    r = ceil(i / c)
    return r, c


def plot(images, cmap="gray", titles=[]):
    r, c = dims(len(images))
    _, ax = plt.subplots(r, c)

    if len(images) > 1:
        ax = ax.ravel()
        for i, img in enumerate(images):
            ax[i].imshow(img, cmap=cmap)
            ax[i].set_axis_off()
            if len(titles) > 0:
                ax[i].set_title(titles[i])
    else:
        ax.imshow(images[0], cmap=cmap)
        ax.set_axis_off()
        if len(titles) > 0:
            ax.set_title(titles[0])
    plt.tight_layout()
    plt.show()
