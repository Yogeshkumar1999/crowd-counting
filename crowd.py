# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:08:33 2019

@author: yogesh
"""

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from scipy import spatial as sp 
import torch
%matplotlib inline
#%%
import torch.nn as nn
'''import torch
from torchvision import models

import h5py

import shutil

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')            

#%%
'''
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  
#%%              
import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    
    
    return img,target
#%%

#%%
#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if (gt_count == 0):
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    print("AETAET",pts.copy())
    leafsize = 2048
    # build kdtree
    print("fasakkkkkk")
    tree = sp.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if (gt_count > 1):
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density
#%%
#set the root to the Shanghai dataset you download
'''
root = 'c:/Users/yogesh/desktop/CSRNet-pytorch-master'
#%%
#now generate the ShanghaiA's ground truth
'''part_A_train = os.path.join(root,'/part_A/train_data/','images/')
part_A_test = os.path.join(root,'/part_A/test_data/','images/')
part_B_train = os.path.join(root,'/part_B/train_data/','images/')
part_B_test = os.path.join(root,'/part_B/test_data/','images/')
'''
part_A_train = "C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_A/train_data/images"
part_A_test = "C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_A/test_data/images"
path_sets = [part_A_train,part_A_test]
#%%
'''
'''
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '//*.jpg')):
        img_paths.append(img_path)'''
#%%
img_paths = []
for img_path in glob.glob("C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_A/train_data/images//*.jpg"):
        img_paths.append(img_path)
for img_path in glob.glob("C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_A/test_data/images//*.jpg"):
        img_paths.append(img_path)

#%%
for img_path in img_paths:
    print (img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if (int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]):
            k[int(gt[i][1]),int(gt[i][0])]=1
    print("shape of k is: ", k.shape)
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
            hf['density'] = k


#%%
#now see a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))
#%%
gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground-truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
#%%
np.sum(groundtruth)# don't mind this slight variation
#%%
#now generate the ShanghaiB's ground truth
'''part_A_train = "C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_B/train_data/images"
part_A_test = "C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_B/test_data/images"
path_sets = [part_B_train,part_B_test]'''
#%%
img_paths = []
for img_path in glob.glob("C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_B/train_data/images//*.jpg"):
        img_paths.append(img_path)
for img_path in glob.glob("C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_B/test_data/images//*.jpg"):
        img_paths.append(img_path)

#%%
'''img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '//*.jpg')):
        img_paths.append(img_path)
'''
#%%
for img_path in img_paths:
    print (img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,15)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
            hf['density'] = k

#%%

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
                       
#%%
img1_paths = []
for img_path in glob.glob("C:/Users/yogesh/Desktop/CSRNet-pytorch-master/ShanghaiTech/part_A/test_data/images//*.jpg"):
        img1_paths.append(img_path)
#%%
model = CSRNet()
#%%
model = model.cuda()
#%%

checkpoint = torch.load('model_best.pth.tar')
#%%
model.load_state_dict(checkpoint['state_dict'])
#%%

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM

import torch
%matplotlib inline
#%%
import tqdm
mae = 0
for i in tqdm(range(len(img_paths))):
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
print (mae/len(img_paths))
#%%
mae = 0
for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    img = img.cuda()
    #img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    print (i,mae)
print (mae/len(img_paths))
#%%










































































































