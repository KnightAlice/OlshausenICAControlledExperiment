from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import scipy.io as sio
from torchvision import datasets, transforms
import torch
import os
import numpy as np

os.makedirs('patches', exist_ok=True)

mat_images = sio.loadmat('./datasets/IMAGES.mat')
imgs = mat_images['IMAGES']

batch_size = 256
n_iters =500
imgsize = 16#9

class Times3:
    def __call__(self, x):
        return x * 3
    
transform = transforms.Compose([
    #FilterSmall(min_size=(imgsize, imgsize)),
    transforms.ToTensor(),
    transforms.RandomCrop(imgsize),
    #Times3(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    #transforms.Lambda(lambda x: 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3])
    #transforms.Lambda(lambda x: x-torch.mean(x))
    #Normalize01()
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None, n_samples=100):
        self.images = images
        self.transform = transform
        self.n_samples = n_samples
        self.num_images = images.shape[2]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_idx = idx % self.num_images
        img = self.images[:,:,img_idx]
        if self.transform:
            img = self.transform(img)
        return img.to(torch.float32)

dataset = ImageDataset(imgs, transform=transform, n_samples=batch_size*n_iters)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

patches = []
# Run simulation
for n, imgs in tqdm.tqdm(enumerate(dataloader)):
    
    '''pre-save data generator'''
    for i in range(imgs.shape[0]):
        x = imgs[i].cpu().numpy() 
        patches.append(x)

patches = np.array(patches)
#patches = patches/ (abs(patches).max() + 1e-8)

np.save('patches/patches_16x16.npy', patches)