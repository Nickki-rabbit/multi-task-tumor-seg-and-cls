# HSI_dataset.py
# File Description: This file is used to create HSI data tuples
#==============================================================================

# import cv2
import torch.utils.data
from scipy.io import loadmat
import numpy as np
import os.path

class LoadDatasetName:
    def __init__(self, data_dir, classes, diagClasses):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        
        self.trainImList = list()
        self.trainAnnotList = list()

        self.valImList = list()
        self.valAnnotList = list()

        self.diagClassTrain = list()
        self.diagClassVal = list()

        self.diagClasses = diagClasses
        self.diagWeights = np.ones(diagClasses, dtype=np.float32)

    def readFile(self, fileName):
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                line_arr = line.split(',')
                img_file = line_arr[0].strip()
                label_file = line_arr[1].strip()
                cls_file = line_arr[2].strip()

                self.trainImList.append(img_file)
                self.trainAnnotList.append(label_file)

                # self.valImList.append(
                self.diagClassTrain.append(cls_file)
                
                # self.valImList.append( # how about train val split
        return 0
    def processData(self):
        print('Processing training data')
        return_val = self.readFile('train.txt')
        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['trainDiag'] = self.diagClassTrain
            data_dict['classWeights'] = self.classWeights
            data_dict['diagClassWeights'] = self.diagWeights

            return data_dict
        return None

   
class HSI_Dataset(torch.utils.data.Dataset):
    """ HSI dataset """
    # flg = 0
    # def __init__(self, imList, gtList, **hyperparams):
    def __init__(self, imList, gtList, diagList):
        self.imList = imList
        self.gtList = gtList
        self.diagList = diagList
        # self.transform = transform
        # self.name = hyperparams['dataset']
        # self.patch_size = hyperparams['patch_size']
        self.patch_size = 16
        self.transform = None
        # self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.flip_augmentation = hyperparams['flip_augmentation']
        # self.radiation_augmentation = hyperparams['radiation_augmentation']
        # self.mixture_augmentation = hyperparams['mixture_augmentation']
        # self.center_pixel = hyperparams['center_pixel']
        # supervision = hyperparams['supervision']

        # fully supervised - use all pixels with label not ignored
        # if supervision == 'full':
        #     mask = np.ones([patch_size, patch_size, 448])
        #     for l in self.ignored_labels:
        #        mask[gt == l] = 0

        # semi-supervised - use all pixels, except padding(?)
        #elif supervision == 'semi':
        #    mask = np.ones_like(gt)
            
    def cut_into_patches(self, data, gt):
        # 0-backgrd, 1-healthy, 2-hcc, 3-icc
        patches = []
        patches_gt = []
        patches_cls_gt = []

        # record the left up corner of tha patches with different classes
        bgrd_flg = []
        healthy_flg = []
        tumor_flg = []
        
        for i in range(data.shape[0] - self.patch_size):
            for j in range(data.shape[1] - self.patch_size):
                # central pixel of a patch
                tt = (gt[i+self.patch_size//2, j+self.patch_size//2]).copy()
                if tt == 0:
                    bgrd_flg.append([i, j])
                elif tt == 1:
                    healthy_flg.append([i, j])
                else:
                    tumor_flg.append([i, j])
        
        bgrd_flg_array = np.array(bgrd_flg) # [num x 2]
        healthy_flg_array = np.array(healthy_flg)
        tumor_flg_array = np.array(tumor_flg)

        # select patches of various classes in random
        random_indices_bgrd = bgrd_flg_array[np.random.choice(bgrd_flg_array.shape[0], size=10, replace=False), ]
        random_indices_healthy = healthy_flg_array[np.random.choice(healthy_flg_array.shape[0], size=10, replace=False), ]
        random_indices_tumor = tumor_flg_array[np.random.choice(tumor_flg_array.shape[0], size=10, replace=False), ]
        # merge patches
        random_patch_indices = np.concatenate((random_indices_bgrd, random_indices_healthy, random_indices_tumor), axis=0) # [N x 2]
        for k in range(random_patch_indices.shape[0]):
            i = random_patch_indices[k, 0]
            j = random_patch_indices[k, 1]
            # generate patch and gt (for seg)  without shuffling
            patch = (data[i:i+self.patch_size, j:j+self.patch_size, :]).copy()
            patch_gt = (gt[i:i+self.patch_size, j:j+self.patch_size]).copy()

            # [for cls] hcc=0 or icc=1, or bgrd=-1
            if np.any(patch_gt == 2):
                patches_cls_gt.append(0)
            elif np.any(patch_gt == 3):
                patches_cls_gt.append(1)
            else:
                patches_cls_gt.append(-1)

            # [for seg] 0-bgrd, 1-healthy, 2-tumor
            patch_gt[patch_gt == 3] = 2
            patches.append(patch)
            patches_gt.append(patch_gt)
        # print('patch shape:', np.shape(patches))
        # print('Image cropped.')

        patches = np.array(patches)
        patches_gt = np.array(patches_gt)
        patches_cls_gt = np.array(patches_cls_gt)
        # to categorical in keras             
        return patches, patches_gt, patches_cls_gt
        
    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx] # specimen_1, 3, 5, ...
        print('the image right now is: ', image_name)
        # patch_img = []
        gt_name = self.gtList[idx]
        img_mat = loadmat('./data_file/specimen.mat')[image_name]
        gt_mat = loadmat('./data_file/specimen_gt.mat')[gt_name]
        cls_mat = self.diagList[idx] # TODO: use diagList in patches
        patch_img_mat, patch_gt_mat, patch_gt_cls_mat = self.cut_into_patches(img_mat, gt_mat) # [30, 16, 16, 448] per hsi image
        
        patch_img = patch_img_mat
        patch_gt = patch_gt_mat
        patch_cls = patch_gt_cls_mat
        # patch_img = np.concatenate((patch_img, img_mat), axis=0)
        
        if self.transform:
            [patch_img, patch_gt] = self.transform(patch_img, patch_gt)
        
        patch_img = np.asarray(np.copy(patch_img).transpose((0, 3, 1, 2)), dtype='float32') # [30, 448, 16, 16]
        patch_gt = np.asarray(np.copy(patch_gt), dtype='int64') # [30, 16, 16]
        patch_cls = np.asarray(np.copy(patch_cls), dtype='int64') #shape? (30,)
        print('PATCH_CLS: ', patch_cls)
        
        patch_img = torch.from_numpy(patch_img)
        patch_gt = torch.from_numpy(patch_gt)
        patch_cls = torch.from_numpy(patch_cls)
        # print('patch_img size is: ', patch_img.shape)
        # print('patch_gt size is: ', patch_gt.shape)
        
        return (patch_img, patch_gt, patch_cls)
