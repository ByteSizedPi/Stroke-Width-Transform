import csv
import os
import re
from random import randint

from skimage.io import imread
from swt import SWT

from plots import plot

directory = (
    "C:/Personal/Coding/Meesters/Scripts/Denoising Tests/1. Generate Images/results"
)

filenames = sorted(
    os.listdir(directory), key=lambda x: int(re.search(r"_index=(\d+)", x).group(1))
)

# Read the CSV file
csv_file = "C:/Personal/Coding/Meesters/Scripts/Denoising Tests/gradient_dir.csv"
csv_data = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        csv_data = [int(x) for x in row]

for i in range(0, 1):
    # idx = randint(0, len(filenames))
    idx = i

    # idx = 3178
    img = imread(os.path.join(directory, filenames[idx]), as_gray=True)

    modded_idx = int(idx / 33)

    grad_dir = 1 if csv_data[modded_idx] else -1

    boxes = SWT(img, 1)
    print(boxes)
