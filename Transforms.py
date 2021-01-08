# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the source code for different types of augmentation and numpy to Tensor Conversion for HS images.
# ==============================================================================

import numpy as np
import torch
import random
import cv2


class Zoom(object): #pass
    """
        Resize the image to the larger size and randomly zoom into an area of the same size as of the original image.
    """

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, label):
        h1, w1 = img.shape[:2]
        startH = random.randint(0, int(abs(self.h - h1) / 2))
        startW = random.randint(0, int(abs(self.w - w1) / 2))
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        img = img[startH:startH + h1, startW:startW + w1]
        label = label[startH:startH + h1, startW:startW + w1]
        return [img, label]


class RandomCropResize(object):
    """
    Randomly crop and resize the given PIL(x) HSI-image with a probability of 0.5
    """

    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            w, h = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]


class RandomHorizontalFlip(object):
    """Randomly flips (horizontally as well as vertically) the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            # image = cv2.flip(image, 0)  # horizontal flip
            image = [np.fliplr(img) for img in image]
            # label = cv2.flip(label, 0)  # horizontal flip
            label = np.fliplr(label)
        if vertical:
            # image = cv2.flip(image, 1)  # veritcal flip
            image = [np.flipud(img) for img in image]
            # label = cv2.flip(label, 1)  # veritcal flip
            label = np.flipud(label)
        return [image, label]


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        for i in range(3):
            image[:, :, i] -= self.mean[i]
        for i in range(3):
            image[:, :, i] /= self.std[i]

        return [image, label]


class ToTensor(object):

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            w, h = label.shape[:2]
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), interpolation=cv2.INTER_NEAREST)

        image = np.array(image)
        image = image.transpose((2, 0, 1)) # permute
        image = image.astype(np.float32)

        image_tensor = torch.from_numpy(image).div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int))  # torch.from_numpy(label)

        return [image_tensor, label_tensor]


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
