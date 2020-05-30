from albumentations import *


def strong_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            MedianBlur(p=1.0, blur_limit=7),
            Blur(p=1.0, blur_limit=7),
            GaussianBlur(p=1.0, blur_limit=7),
        ], p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(p=1.0),
            GaussNoise(p=1.0),
        ], p=1),
        OneOf([
            ElasticTransform(p=1.0, alpha=1.0, sigma=30, alpha_affine=20),
            GridDistortion(p=1.0, num_steps=5, distort_limit=0.3),
            OpticalDistortion(p=1.0, distort_limit=0.5, shift_limit=0.5)
        ], p=0.5)
    ],  p=p)


def simple_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RGBShift(p=0.2, r_shift_limit=(-5, 5), g_shift_limit=(-5, 5), b_shift_limit=(-5, 5)),
    ],  p=p)


