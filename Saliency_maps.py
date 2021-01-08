#File Description: This file is used to visualize the saliency maps.
#==================================================================

import numpy as np
import torch
from torch.autograd import Variable
#import glob
from scipy.io import loadmat
from scipy.io import savemat
import cv2
import sys
# sys.path.insert(0, '../stage2/')
#import Model_stage2 as Net
import Model_stage2 as Net
import os
from PIL import Image
import skimage.io as io

from models_add_unet import Y_Net
import matplotlib.pyplot as plt

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
            
model = Y_Net(3, 3,119)
# model.load_state_dict(torch.load('./pretrained_Y1/model_99_12052103.pth'))
model.load_state_dict(torch.load('./results_ynet_add_trans_Y1/model_99.pth'))

# tell pytorch not to compute grad for model params
for param in model.parameters():
    param.requires_grad = False


patch_size = 64
model = model.cuda()
model.eval()

image_list = loadmat('./data_file/specimen_size.mat')
gt_list = loadmat('./data_file/specimen_gt.mat')

out_dir = './results_small_Ynet/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

def compute_saliency_maps(X, y, model):
    """
    - X: input images: Tensor of shape (N, C, H, W)
    - y: label for X: LongTensor of shape (N,) || y should be non-tumor(-1 or 2), hcc(0), icc(1)
    - model: A pretrained CNN / UNet that will be used to compute the saliency map
    
    Return: 
    - saliency : A Tensor of shape (N, H, W) giving the salitncy maps for the input images
    """
    
    # test mode
    model.eval()
    # now the X needs gradient
    X.requires_grad_()
    
    saliency = None
    
    # Y_Net has two outputs: one is the seg_map, one is the cls result-->logits
    seg_map, logits = model.forward(X)
    print('the cls output: ', logits)
    # use gather to select one entry from each row in pytorch

    
    # loss1 = logits.gather(1, y.view(-1, 1)).squeeze() # get the correct label of image
    # loss1.backward()
    
    loss_func = torch.nn.CrossEntropyLoss()
    y = y.reshape(-1)
    loss = loss_func(seg_map, y)
    loss.backward()
    
    # loss.backward() #??? only compute loss of pixels correctly classified
    
    saliency = abs(X.grad.data) # the abs of X's grad - 1x448x64x64
    
    spec_saliency = saliency.squeeze().cpu().numpy()
    spec_saliency_mat = {'spec_saliency': spec_saliency}
    savemat('./saliency_map/spec_saliency.mat', spec_saliency_mat)
    saliency, spec_idx = torch.max(saliency, dim=1) # saliency & spec_idx : 1x64x64
    return saliency.squeeze(), spec_idx.squeeze()
    
#for imgName in [1,2,3,4,5,6,7,9,10,
#                13,14,15,16,17,18,19,20,
#                25,28,29,30,34,36,37,41,
#                'inv_1','inv_2','inv_3','inv_4','inv_5','inv_6','inv_7',
#                'inv_10','inv_16','inv_17','inv_18','inv_19','inv_20',
#                'inv_25','inv_28','inv_29','inv_30','inv_34','inv_36','inv_37','inv_41']:
# X = [30, 34]
for imgName in [20]:   # do it in all image domain
    img = image_list['sample_{}'.format(imgName)].astype(np.float32)
    idx1 = [n for n in range(40)]
    idx7 = [n for n in range(157, 192)]
    idx8 = [n for n in range(404, 448)]
    idx = idx1+idx7+idx8
    img = (img[0:256:4, 0:320:5, idx]).copy()
    # img = (img[0:128:2, 0:128:2, :]).copy()
    # img = (img[10:138:2, 15:143:2, :]).copy()
    
    # X = img.copy()
    # X = (img[64:128, 64:128, :]).copy()
    # X = (img[0:64, 0:64, :]).copy()
    # X = (img[0:64, 64:128, :]).copy()
    
    # X = X.transpose((2, 0, 1))
    # X_tensor = torch.from_numpy(X)
    # X_tensor = torch.unsqueeze(X_tensor, 0) # 1x448x192x192
    # X_var = Variable(X_tensor).cuda()
    
    saliency_output = torch.zeros((img.shape[0], img.shape[1]))
    #cls_output = []
    
    y = [0]
    y_tensor = torch.LongTensor(y)
    y_tensor = torch.unsqueeze(y_tensor, 0)
    y_var = Variable(y_tensor).cuda()    
    
    for i in range(0, img.shape[0] - patch_size+1, patch_size):
        for j in range(0, img.shape[1] - patch_size+1, patch_size):
            print('i is ', i, 'j is ', j)
            patch = (img[i:i+patch_size, j:j+patch_size, :]).copy() # 64 * 64 * 448
            patch = patch.transpose((2, 0, 1)) #448*64*64
            patch_tensor = torch.from_numpy(patch)
            patch_tensor = torch.unsqueeze(patch_tensor, 0) # 1*448*64*64 - add a batch dim
            patch_var = Variable(patch_tensor).cuda()
            
            # img_out, cls_out = model(patch_var) # 1*3*64*64,
            
            saliency, spec_idx = compute_saliency_maps(patch_var, y_var, model) # 1x64x64
            # saliency = torch.squeeze(saliency, 0) # 64x64
            saliency_output[i:i+patch_size, j:j+patch_size] = saliency
            
    saliency_output = saliency_output.cpu().numpy()
    spec_idx = spec_idx.cpu().numpy()
    

    # tumor_spec
    # import pandas as pd
    
    spec_idx_mat = {'histo_spec': spec_idx}
    savemat('./saliency_map/spec_idx.mat', spec_idx_mat)
    
    # img_spec = pd.Series(spec_idx.reshape(-1,1).squeeze())
    # img_spec.plot.hist()
    
    
    # N = X.shape[0]
    # io.imsave(os.path.join('./',"saliency_output_sample_30.png"), saliency_output)
    N = 1
    for i in range(N):
        plt.subplot(2, N, i+1)
        plt.imshow(img[:, :, 250])
        plt.axis('off')
        # plt.title(class_name[y[i]])
        plt.subplot(2, N, N+i+1)
        plt.imshow(saliency_output, cmap=plt.cm.hot)
        # plt.imshow(saliency_output)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)

    plt.show()
    #        img_out_norm = torch.squeeze(img_out, 0) # 3*64*64   - remove the batch dim
            
    #       output[:, i:i+patch_size, j:j+patch_size] = img_out_norm
    #       cls_output.append(cls_out.detach().cpu().tolist())
        
        
#    cls_output = np.array(cls_output) # patch x 1 x 3

#    diagClass = np.squeeze(np.argmax(cls_output, axis=2))
#        
#    from collections import Counter
#    patch_cls_count = Counter(diagClass)
#    
#    if patch_cls_count[0] > patch_cls_count[1]:
#        diag_output = 0
##        print('The type of liver tumor in ', str(imgName), ' is HCC.')
#    elif patch_cls_count[0] < patch_cls_count[1]:
#        diag_output = 1
#        print('The type of liver tumor in ', str(imgName), ' is ICC.')
#    else:
#        diag_output = -1
#        print(str(imgName), ': Unknown type')
        
#    prob, classMap = torch.max(output, 0)
#    classMap_numpy = classMap.data.cpu().numpy()
    
#    im_pil = Image.fromarray(np.uint8(classMap_numpy))
#    im_pil.putpalette(pallete)
#    
#    im_pil.save(out_dir + str(imgName) + '.png')

