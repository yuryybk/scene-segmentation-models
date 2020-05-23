import random
import itertools
import cv2
import data_loader
import numpy as np


def segmentation_generator(images_path,
                           mask_path,
                           batch_size,
                           n_classes,
                           input_height,
                           input_width,
                           output_height,
                           output_width):
    img_seg_pairs = data_loader.get_pairs_from_paths(images_path, mask_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            X.append(
                data_loader.get_image_array(im, input_width, input_height, ordering="channel_last")
            )

            Y.append(
                data_loader.get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
            )

        yield np.array(X), np.array(Y)
