#File Description: This file is used to visualize the segmentation masks
#==================================================================

import numpy as np
import torch
from torch.autograd import Variable
#import glob
from scipy.io import loadmat
import cv2
import sys
# sys.path.insert(0, '../stage2/')
#import Model_stage2 as Net
import Model_stage2 as Net
import os
from PIL import Image
import skimage.io as io

from models_add_unet import Y_Net
import segmentation_models_pytorch as smp

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

#model = Net.ResNetC1_YNet(3, 3)
# model = Y_Net(3, 3, 448)


aux_params = dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.25,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=3,                 # define number of output labels
)
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    #encoder_weigths=torch.load('./results_ynet_add_trans_Custom_block/model_100.pth'),
    in_channels=448,
    classes=3,
    aux_params=aux_params,
)
model.load_state_dict(torch.load('./results_ynet_add_trans_Custom_block/model_100.pth'))
# model.load_state_dict(torch.load('../stage2/pretrained_model_st2/ynet_c1.pth'))
            

# model = Y_Net(3, 3)
# model.load_state_dict(torch.load('./pretrained_Y1/model_99_12052103.pth'))

patch_size = 64
model = model.cuda()
model.eval()

#image_list = glob.glob('./data_file/*.png')
image_list = loadmat('./data_file/specimen_49.mat')
gt_list = loadmat('./data_file/specimen_gt.mat')

# gth_dir = '../stage1/data/valannot/'
# out_dir = './results_select_bands/'
# out_dir = './results_C1/'
# out_dir = './results_u1/'
out_dir = './results_custom/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
#for imgName in [1, 3, 5, 8, 10, 14, 15, 16]:
for imgName in [1,2,3,4,5,6,7,8,9,10,
                13,14,15,16,17,18,19,20,
                25,28,29,30,34,36,37,41,
                'inv_1','inv_2','inv_3','inv_4','inv_5','inv_6','inv_7',
                'inv_10','inv_16','inv_17','inv_18','inv_19','inv_20',
                'inv_25','inv_28','inv_29','inv_30','inv_34','inv_36','inv_37','inv_41']:
    # img = cv2.imread(imgName).astype(np.float32)
    img = image_list['sample_{}'.format(imgName)].astype(np.float32)
    # img /=255 # do no need this for mat data
    output = torch.zeros((3, img.shape[0], img.shape[1]))
    cls_output = []

    # mask
    idx1 = [ n for n in range(40)]
    idx2 = [ n for n in range(115,187)]
    idx3 = [n for n in range(221, 241)]
    idx4 = [n for n in range(262, 272)]
    idx5 = [n for n in range(337,373)]
    idx6 = [n for n in range(418,448)]

    idx7 = [n for n in range(157, 192)]
    idx8 = [n for n in range(404, 448)]
    # idx = idx1+idx2+idx3+idx4+idx5+idx6
    idx = idx1 + idx7 + idx8
    # idx = [72, 289, 265, 317, 227, 207, 171, 325, 137, 154]
    # img = img[:,:,idx]
    
    for i in range(0, img.shape[0] - patch_size+1, patch_size):
        for j in range(0, img.shape[1] - patch_size+1, patch_size):
            patch = (img[i:i+patch_size, j:j+patch_size, :]).copy() # 64 * 64 * 448
            patch = patch.transpose((2, 0, 1)) #448*64*64
            patch_tensor = torch.from_numpy(patch)
            patch_tensor = torch.unsqueeze(patch_tensor, 0) # 1*448*64*64 - add a batch dim
            patch_var = Variable(patch_tensor).cuda()
            
            img_out, cls_out = model(patch_var) # 1*3*64*64,
            
            img_out_norm = torch.squeeze(img_out, 0) # 3*64*64   - remove the batch dim
            #if img.shape[0]-i < patch_size or img.shape[1]-j < patch_size:
            #    output[:, i:i+patch_size, j:j+patch_size] = 
            output[:, i:i+patch_size, j:j+patch_size] = img_out_norm
            cls_output.append(cls_out.detach().cpu().tolist())

    cls_output = np.array(cls_output) # patch x 1 x 3
    
    diagClass = np.squeeze(np.argmax(cls_output, axis=2))
        
    from collections import Counter
    patch_cls_count = Counter(diagClass)
    
    if patch_cls_count[0] > patch_cls_count[1]:
        diag_output = 0
        print('The type of liver tumor in ', str(imgName), ' is HCC.')
    elif patch_cls_count[0] < patch_cls_count[1]:
        diag_output = 1
        print('The type of liver tumor in ', str(imgName), ' is ICC.')
    else:
        diag_output = -1
        print(str(imgName), ': Unknown type')
        
    prob, classMap = torch.max(output, 0)
    classMap_numpy = classMap.data.cpu().numpy()
    
    im_pil = Image.fromarray(np.uint8(classMap_numpy))
    im_pil.putpalette(pallete)
    
    # name = imgName.split('/')[-1]
    
    im_pil.save(out_dir + str(imgName) + '.png')

    #gth = cv2.imread(gth_dir + os.sep + imgName.split(os.sep)[-1], 0).astype(np.uint8)
    #gth = Image.fromarray(gth)
    #gth.putpalette(pallete)
    #gth.save(out_dir + 'gth_' + name)

