import matplotlib
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from data_loader import *
from nyu_v2_descriptor import *
from augmentations import *
matplotlib.use('TkAgg')


def augment_and_show(aug, image, mask=None, aug_count=1):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f, ax = plt.subplots(aug_count + 1, 2, figsize=(16, 16))
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original image')

    ax[0, 1].imshow(mask, interpolation='nearest')
    ax[0, 1].set_title('Original mask')

    for i in range(1, aug_count + 1):

        augmented = aug(image=image, mask=mask)
        image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)

        ax[i, 0].imshow(image_aug)
        ax[i, 0].set_title('Augmented image')

        ax[i, 1].imshow(mask_aug, interpolation='nearest')
        ax[i, 1].set_title('Augmented mask')

    f.tight_layout()
    plt.show()


def test_augmentation_nyu2():
    data = NYU2Data()
    image_path, mask_path = get_first_image_pair_from_path(data.get_train_rgb_path(), data.get_train_mask_path())
    aug = simple_aug(p=1)
    image = cv2.imread(image_path, 1)
    mask = cv2.imread(mask_path, 1)
    augment_and_show(aug, image, mask, 5)


test_augmentation_nyu2()
