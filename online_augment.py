import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    RGBShift,
    HueSaturationValue,
    ChannelShuffle,
    Blur,
    MedianBlur,
    JpegCompression,
)


def visualize(image, mask, original_image=None, original_mask=None):
    numpy_horizontal = np.hstack((image, mask))
    cv2.imshow("hor", numpy_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment(image, mask):
    original_height, original_width = image.shape[:2]
    aug = PadIfNeeded(p=1, min_height=128, min_width=128)
    augmented = aug(image=image, mask=mask)

    image_padded = augmented['image']
    mask_padded = augmented['mask']

    aug = Compose([
        OneOf([RandomSizedCrop(p=0.5, min_max_height=(int(original_height/4), int(original_height/2)),
                              height=original_height, width=original_width),
               PadIfNeeded(min_height=original_height,
                           min_width=original_width, p=0.5)], p=1),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * random.uniform(0.07,0.2),
                             alpha_affine=120 * random.uniform(0.03,0.5)),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=0.2, shift_limit=0.2)
        ], p=0.8),
        CLAHE(p=0.8),
        RandomContrast(p=0.8),
        RandomBrightness(p=0.8),
        RandomGamma(p=0.8),
        RGBShift(p=0.1),
        HueSaturationValue(p=0.1),
        ChannelShuffle(p=0.1),
        Blur(p=0.3),
        MedianBlur(p=0.3),
        JpegCompression(p=0.8)
    ])

    augmented = aug(image=image_padded, mask=mask_padded)
    image_v = augmented['image']
    mask_v = augmented['mask']

    aug = PadIfNeeded(p=1, min_height=1024, min_width=1024)
    augmented = aug(image=image_v, mask=mask_v)

    image_v = augmented['image']
    mask_v = augmented['mask']

    # image_v = cv2.resize(image_v, (64, 64))
    # mask_v = cv2.resize(image_v, (64, 64))
    return image_v, mask_v
    # aug = PadIfNeeded(p=1, min_height=1000, min_width=1000)
    # augmented = aug(image=image_v, mask=mask_v)
    #
    # image_padded = augmented['image']
    # mask_padded = augmented['mask']
    # return image_padded, mask_padded

if __name__ == "__main__":
    image = cv2.imread('photos/0001.jpg')
    mask = cv2.imread('real_mask/0001.png')

    image, mask = augment(image, mask)
    visualize(image, mask)