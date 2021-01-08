import torch
import torch.utils
import torch.utils.data
import numpy as np
import os


class flip(object):
    """
    Randomly flip the HSI images horizontally and vertically with a probability of 0.5.
    """
    def __call__(self, image, label, label1): # image(64,64,448)
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        # print('doing the flip right now')
        if horizontal:
            # print('horizontal')
            image = np.fliplr(image).copy() # or numpy array will have negative strides
            label = np.fliplr(label).copy()
        if vertical:
            # print('vertical')
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        return [image, label, label1]

class radiation_noise(object):
    """
    Add radiation noise onto HSI data.
    """
    def __call__(self, image, label, label1):
        alpha_range=(0.9, 1.1)
        beta=1/25
        
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=image.shape)
        image = alpha * image + beta * noise
        
        return [image, label, label1]

class ToTensor(object):
    
    # def __init__(self, scale=1):
    #     self.scale = scale

    def __call__(self, image, label, label1):
        image = np.array(image).reshape(-1, 64, 64, 448)
        label = np.array(label).reshape(-1, 64, 64)
        label1 = np.array(label1)
        
        image = np.asarray(np.copy(image).transpose((0, 3, 1, 2)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')
        label1 = np.asarray(np.copy(label1), dtype='int64')
        # image = image.astype(np.float32)

        print('image shape is: ', image.shape)
        print('label shape is: ', gt.shape)
        image_tensor = torch.from_numpy(image)
        # label_tensor = torch.LoneTensor(np.array(labelm, dtype=np.int))
        label_tensor = torch.from_numpy(label)
        label1_tensor = torch.from_numpy(label1)
        
        return [image_tensor, label_tensor, label1_tensor]

class Compose(object):
    """Compose several transforms together
    """    
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
            return args
