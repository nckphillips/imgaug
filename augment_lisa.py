from imgaug import augmenters as iaa
import numpy as np
import cv2
import os

CLASS_DIRS = ["./work_folder/00001/", "./work_folder/00002/", "./work_folder/00004/", "./work_folder/00014/"]
files = []

for c in CLASS_DIRS:
    lof = os.listdir(c)
    for i in range(0, len(lof)): lof[i] = c + lof[i]
    files = lof
images = []
np.array(images)
for filename in files:
    img = cv2.imread(filename)
    images.append(img)

"""Augment the images"""
sometimes = lambda aug: iaa.Sometimes(.5, aug)

augmentation_seq = iaa.Sequential([

    sometimes(iaa.Salt(0.03)),
    sometimes(iaa.SaltAndPepper(0.03)),
    sometimes(iaa.CoarseDropout(size_percent=.3)),
    sometimes(iaa.GaussianBlur(sigma=1))
])

images = augmentation_seq.augment_images(images)

i = 0
for filename in files:
    #filename.replace(".ppm", ".aug.ppm")
    filename = filename[:-3]
    filename += 'aug.ppm'
    print(filename)

    cv2.imwrite(filename, images[i])
    i += 1
