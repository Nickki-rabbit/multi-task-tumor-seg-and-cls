# HSI_dataset.py
# File Description: This file is used to create HSI data tuples
#==============================================================================

# import cv2
import torch.utils.data
from scipy.io import loadmat
import numpy as np
import os.path

class LoadDatasetPatch:
    def __init__(self, data_dir, classes, diagClasses, patch_size):
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

        self.patch_size = patch_size

    def readFile(self, fileName, train_flg=False):
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                line_arr = line.split(',')
                img_file = line_arr[0].strip()
                label_file = line_arr[1].strip()
                # cls_file = line_arr[2].strip() # no use for now
                if train_flg == True:
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                    # self.diagClassTrain.append(cls_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)
                # self.diagClassVal.append(cls_file)
        return 0
    
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

            # [for cls] hcc=0 or icc=1, or bgrd=2
            if np.any(patch_gt == 2):
                patches_cls_gt.append(0)
            elif np.any(patch_gt == 3):
                patches_cls_gt.append(1)
            else:
                patches_cls_gt.append(2)

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
    
    def norm_data(self,specimen_data):
        data = specimen_data
        for i in range(data.shape[-1]):
            data[:,:,i] = (data[:,:,i] - np.mean(data[:,:,i])) / np.std(data[:,:,i])

        return data
                
    def processData(self, Train_flg=False):
        return_val = return_val1 = 0
        if Train_flg == True:
            print('Processing training data')
            return_val = self.readFile('train.txt', True)
        else:
            print('Processing the validation data')
            return_val1 = self.readFile('val.txt', False)
        
        # print('Pickling data')
        if return_val == 0 and return_val1 == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['trainDiag'] = self.diagClassTrain
            
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList
            data_dict['valDiag'] = self.diagClassVal
            
            data_dict['classWeights'] = self.classWeights
            data_dict['diagClassWeights'] = self.diagWeights

            
            if Train_flg == True: # training set
                ### patch part
                img0 = loadmat('./data_file/specimen_49.mat')[data_dict['trainIm'][0]]
                # img0 = self.norm_data(img0)
                gt0 = loadmat('./data_file/specimen_49_gt.mat')[data_dict['trainAnnot'][0]]
                patch_img, patch_gt, patch_cls = self.cut_into_patches(img0, gt0)
                for i in range(1, len(self.trainImList)):
                    img_mat = loadmat('./data_file/specimen_49.mat')[data_dict['trainIm'][i]]
                    # img_mat = self.norm_data(img_mat)
                    gt_mat = loadmat('./data_file/specimen_49_gt.mat')[data_dict['trainAnnot'][i]]
                    patch, gt, cls = self.cut_into_patches(img_mat, gt_mat)
                    # patch shape: (30, 64, 64, 448)
                    patch_img = np.concatenate((patch_img, patch), axis=0) # (30*6,64,64,448)
                    patch_gt = np.concatenate((patch_gt, gt), axis=0) # (30*6,64,64)
                    patch_cls = np.concatenate((patch_cls, cls), axis=0) # (30*6,) dtype=uint8
            else: # validating set
                img0 = loadmat('./data_file/specimen_49.mat')[data_dict['valIm'][0]]
                gt0 = loadmat('./data_file/specimen_49_gt.mat')[data_dict['valAnnot'][0]]
                patch_img, patch_gt, patch_cls = self.cut_into_patches(img0, gt0)
                for i in range(1, len(self.valImList)):
                    img_mat = loadmat('./data_file/specimen_49.mat')[data_dict['valIm'][i]]
                    gt_mat = loadmat('./data_file/specimen_49_gt.mat')[data_dict['valAnnot'][i]]
                    patch, gt, cls = self.cut_into_patches(img_mat, gt_mat)
                    patch_img = np.concatenate((patch_img, patch), axis=0) # (30*6,64,64,448)
                    patch_gt = np.concatenate((patch_gt, gt), axis=0) # (30*6,64,64)
                    patch_cls = np.concatenate((patch_cls, cls), axis=0) # (30*6,) dtype=uint8
            
            return patch_img, patch_gt, patch_cls, data_dict
        return None

   
class HSI_Dataset(torch.utils.data.Dataset):
    """ HSI dataset """
    # flg = 0
    # def __init__(self, imList, gtList, **hyperparams):
    # def __init__(self, im, gt, clsList):
    def __init__(self, im, gt, clsList, patch_size, transform=None):
        self.patch_img = im
        self.patch_gt = gt
        self.patch_cls = clsList
        self.transform = transform
        # self.name = hyperparams['dataset']
        # self.patch_size = hyperparams['patch_size']
        self.patch_size = patch_size
        # self.transform = None
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
        
    def __len__(self):
        return len(self.patch_img) # all 180

    def __getitem__(self, idx):
        patch_img = self.patch_img[idx] # (180,64,64,448) -> (64, 64, 448)
        patch_gt = self.patch_gt[idx] # (64, 64)
        patch_cls = self.patch_cls[idx] # ()
        
        ## original ones
        # patch_img = (patch_img).copy().reshape(-1, self.patch_size, self.patch_size, 448)
        # patch_gt = (patch_gt).copy().reshape(-1, self.patch_size, self.patch_size)
        # patch_img = np.asarray(np.copy(patch_img).transpose((0, 3, 1, 2)), dtype='float32') # [30, 448, 16, 16]
        # patch_img = np.asarray(np.copy(patch_img), dtype='float32')
        # for i in range(patch_img.shape[-1]):
        patch_img = np.asarray(np.copy(patch_img), dtype='float32')
        patch_gt = np.asarray(np.copy(patch_gt), dtype='int64') 
        patch_cls = np.asarray(np.copy(patch_cls), dtype='int64') 
            
        channel_shape = patch_img.shape[-1]
        
        #print('PATCH_IMG before trans SHAPE is ', patch_img.shape) # (64, 64, 448)
        if self.transform:
            [patch_img, patch_gt, patch_cls] = self.transform(patch_img, patch_gt, patch_cls)
            
        patch_img = np.asarray(np.copy(patch_img).reshape(-1,self.patch_size,self.patch_size,channel_shape).transpose((0,3,1,2)))
        patch_gt = patch_gt.copy().reshape(-1,self.patch_size,self.patch_size) # maybe no need
        patch_img = torch.from_numpy(patch_img)
        patch_gt = torch.from_numpy(patch_gt)
        patch_cls = torch.from_numpy(patch_cls)
        #print('patch_img size is: ', patch_img.shape) # (1, 448, 64, 64)
        #print('patch_gt size is: ', patch_gt.shape)

        ### return(patch_img, path_img_val, patch_gt, patch_gt_val, patch_cls, patch_cls_val)
        return (patch_img, patch_gt, patch_cls)
    
