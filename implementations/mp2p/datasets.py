import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        
        #img_B = img.crop((0, 0, w/2, h))
        #img_A = img.crop((w/2, 0, w, h))
        
        img_B = img.crop((0, 0, w/4, h))
        img_A = img.crop((w/4, 0, 2*(w/4), h))
        img_C = img.crop((2*(w/4), 0, 3*(w/4), h))
        img_D = img.crop((3*(w/4), 0, w, h))

        if np.random.random() < 0.5:
            if img.mode == 'RGB':
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
                img_C = Image.fromarray(np.array(img_C)[:, ::-1, :], 'RGB')
            elif img.mode == 'L':
                img_A = Image.fromarray(np.array(img_A)[:, ::-1], 'L')
                img_B = Image.fromarray(np.array(img_B)[:, ::-1], 'L')
                img_C = Image.fromarray(np.array(img_C)[:, ::-1], 'L')
                img_D = Image.fromarray(np.array(img_D)[:, ::-1], 'L')
                img_C.point(lambda x: x > 0.9 and 1)
                img_D.point(lambda x: x > 0.9 and 1)
                

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        img_C = self.transform(img_C)
        img_D = self.transform(img_D)

        
        #return {'A': img_A, 'B': img_B}
        return {'A': img_A, 'B': img_B, 'C': img_C, 'D': img_D}

    def __len__(self):
        return len(self.files)
