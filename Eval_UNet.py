#File Description: This file is used to visualize the segmentation masks
#=================================================================

import numpy as np
import torch
from torch.autograd import Variable
#import glob
from scipy.io import loadmat
import cv2
import sys
# sys.path.insert(0, '../stage2/')
#import Model_stage2 as Net
import Model_modified as Net
import os
from PIL import Image
import skimage.io as io

from models_add_unet import Y_Net

# pallete for 3 classes: bcakgrd, liver, tumor
pallete = [
#             255, 255, 255,
#            130, 0, 130,
            0, 0, 130,
#            255, 150, 255,
            150 ,150 ,255,
#            0 ,255 ,0,
#            255, 255 ,0,
            255, 0, 0]

#model = Net.ResNetC1_YNet(3, 2)
#model.load_state_dict(torch.load('./results_ynet_C1/model_99.pth'))
# model.load_state_dict(torch.load('../stage2/pretrained_model_st2/ynet_c1.pth'))
            
#model = Net.ResNetC1(3)
#model.load_state_dict(torch.load('./results_C1/model_99.pth'))

model = Y_Net(3)
model.load_state_dict(torch.load('./results_U1/model_99.pth'))

patch_size = 16
model = model.cuda()
model.eval()

#image_list = glob.glob('./data_file/*.png')
image_list = loadmat('./data_file/specimen.mat')
gt_list = loadmat('./data_file/specimen_gt.mat')

# gth_dir = '../stage1/data/valannot/'
#out_dir = './results_ynet_c1/'
#out_dir = './results_c1/'
out_dir = './results_u1/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
#for imgName in [1, 3, 5, 8, 10, 14, 15, 16]:
#for imgName in [2, 4, 5, 10, 13, 14, 15, 16, 20, 29, 'inv_6', 'inv_7', 'inv_16', 'inv_20','inv_20', 'inv_29']:
for imgName in [1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,25,28,29,
                'inv_1','inv_2','inv_3','inv_4','inv_5','inv_6','inv_7',
                'inv_10','inv_16','inv_17','inv_18','inv_19',
                'inv_20','inv_25','inv_28','inv_29']:
    # img = cv2.imread(imgName).astype(np.float32)
    img = image_list['sample_{}'.format(imgName)].astype(np.float32)
    # img /=255 # do no need this for mat data
    output = torch.zeros((3, img.shape[0], img.shape[1]))
    cls_output = []
    
    for i in range(0, img.shape[0] - patch_size, patch_size):
        for j in range(0, img.shape[1] - patch_size, patch_size):
            patch = (img[i:i+patch_size, j:j+patch_size, :]).copy() # 64 * 64 * 448
            #patch = np.reshape(patch, (-1, patch_size, patch_size, 448)) # 1*64*64*448
            patch = patch.transpose((2, 0, 1)) #448*64*64
            patch_tensor = torch.from_numpy(patch)
            patch_tensor = torch.unsqueeze(patch_tensor, 0) # 1*448*64*64
            patch_var = Variable(patch_tensor).cuda()
            img_out = model(patch_var)
            #img_out, cls_out = model(patch_var) # 1*3*64*64
            img_out_norm = torch.squeeze(img_out, 0) # 3*64*64
            output[:, i:i+patch_size, j:j+patch_size] = img_out_norm
            #cls_output.append(cls_out.detach().cpu().tolist())

    
    
    #cls_output = np.array(cls_output) # patch x 1 x 2
    #cls_output = np.squeeze(np.argmax(cls_output, axis=2))
    
        
    from collections import Counter
    patch_cls_count = Counter(cls_output)
    if patch_cls_count[0] > patch_cls_count[1]:
        diag_output = 0
        print('The liver tumor type of ', str(imgName), ' is HCC.')
    elif patch_cls_count[0] < patch_cls_count[1]:
        diag_output = 1
        print('THe liver tumor type of ', str(imgName), ' is ICC.')
    else:
        diag_output = -1
        print(str(imgName), ': Unknown type')
        
    prob, classMap = torch.max(output, 0)
    classMap_numpy = classMap.data.cpu().numpy()
    
    # img = img.transpose((2,0,1))
    # img_tensor = torch.from_numpy(img)
    # img_tensor = torch.unsqueeze(img_tensor, 0) # add a batch dimension
    # img_variable = Variable(img_tensor).cuda()
    # img_out, sal_out = model(img_variable)
    
    # remove the batch dimension    
    # img_out_norm = torch.squeeze(img_out, 0)
    # prob, classMap = torch.max(img_out_norm, 0)
    # classMap_numpy = classMap.data.cpu().numpy()

    # output = img_out_norm.detach().cpu().numpy()
    # io.imsave(os.path.join(out_dir, 'segmented.png'), output.transpose((1, 2, 0)))
    im_pil = Image.fromarray(np.uint8(classMap_numpy))
    im_pil.putpalette(pallete)

    
    # name = imgName.split('/')[-1]
    
    im_pil.save(out_dir + str(imgName) + '.png')

    #gth = cv2.imread(gth_dir + os.sep + imgName.split(os.sep)[-1], 0).astype(np.uint8)
    #gth = Image.fromarray(gth)
    #gth.putpalette(pallete)
    #gth.save(out_dir + 'gth_' + name)

