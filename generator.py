import random
import itertools
import cv2
import data_loader
import numpy as np
from augmentations import *


def segmentation_generator(images_path,
                           mask_path,
                           batch_size,
                           n_classes,
                           input_height,
                           input_width,
                           output_height,
                           output_width,
                           do_augment):
    img_seg_pairs = data_loader.get_pairs_from_paths(images_path, mask_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            image, mask = next(zipped)

            image = cv2.imread(image, 1)
            mask = cv2.imread(mask, 1)

            if do_augment:
                aug = strong_aug(p=0.5)
                augmented = aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            X.append(
                data_loader.get_image_array(image, input_width, input_height, ordering="channel_last")
            )

            Y.append(
                data_loader.get_segmentation_array(mask, n_classes, output_width, output_height, no_reshape=True)
            )

        yield np.array(X), np.array(Y)
