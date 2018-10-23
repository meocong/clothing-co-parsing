import glob
import scipy.io
import cv2
import numpy as np
import random
import os

pixel_level_files = glob.glob("./real_mask/*.png")

for mask in pixel_level_files:
    photo = mask.replace("real_mask", "photos").replace("png","jpg")

    img = cv2.imread(photo)
    imask = cv2.imread(mask)

    img_name = photo.split("/")[-1].replace("jpg","png")
    if (random.randint(0,100) < 5):
        cv2.imwrite("original_data/test/" + img_name, img)
        cv2.imwrite("original_data/val/" + img_name, img)
        cv2.imwrite("original_data/test_labels/" + img_name.replace(".png","_L.png"), imask)
        cv2.imwrite("original_data/val_labels/" + img_name.replace(".png","_L.png"), imask)
        # os.system("cp {0} {1}".format(photo, "original_data/test/" + img_name))
        # os.system("cp {0} {1}".format(photo, "original_data/val/" + img_name))
        #
        # os.system("cp {0} {1}".format(mask, "original_data/test_labels/" + img_name.replace("png","_L.png")))
        # os.system("cp {0} {1}".format(mask, "original_data/val_labels/" + img_name.replace("png","_L.png")))
    else:
        # os.system("cp {0} {1}".format(photo, "original_data/train/" + img_name))
        # os.system("cp {0} {1}".format(mask, "original_data/train_labels/" + img_name.replace("png","_L.png")))

        cv2.imwrite("original_data/train/" + img_name, img)
        cv2.imwrite("original_data/train_labels/" + img_name.replace(".png", "_L.png"),imask)
        pass
