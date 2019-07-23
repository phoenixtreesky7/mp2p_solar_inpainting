import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="kernel_radius_1", help='name of the dataset')
parser.add_argument('--datasave_name', type=str, default="mp2p_k1_l01", help='save of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=10, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)



os.makedirs('images/%s' % opt.datasave_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.datasave_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_loss_D = 1

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

# Initialize generator and discriminator
generator = GeneratorUNet()
print('it is generator')
print(generator)

discriminator = Discriminator()
print('it is discriminator')
print(discriminator)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.datasave_name, opt.epoch)))
    discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (opt.datasave_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) ]

dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode='val2'),
                            batch_size=1, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['B'].type(Tensor))
    real_B = Variable(imgs['A'].type(Tensor))
    real_C = Variable(imgs['C'].type(Tensor))
    real_C = real_C.clamp(0, 1)
     
    fake_B = generator(real_A)
    fake_B = fake_B.mul(real_C) + real_A.mul(1-real_C)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'images/%s/%s.png' % (opt.datasave_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

#prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        #real_A = Variable(batch['B'].type(Tensor))
        #real_B = Variable(batch['A'].type(Tensor))
        real_A = Variable(batch['B'].type(Tensor))
        real_B = Variable(batch['A'].type(Tensor))
        real_C = Variable(batch['C'].type(Tensor))
        real_D = Variable(batch['D'].type(Tensor))
        real_C = real_C.clamp(0, 1)
        real_D = real_D.clamp(0, 1)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        #real_AC = real_A.mul(real_C)
        real_AD = real_A.mul(real_D)
        
        real_BC = real_B.mul(real_C)
        real_BD = real_B.mul(real_D)

        real_BCD = real_BD.mul(real_D)
        # GAN loss
        fake_B = generator(real_A)
        fake_B = fake_B.mul(real_C) + real_A.mul(1-real_C)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)


        # GAN loss D
        fake_BD = fake_B.mul(real_D)
        pred_fake_D = discriminator(fake_BD, real_AD)
        loss_GAN_D = criterion_GAN(pred_fake_D, valid)

        # Pixel-wise loss C
        loss_pixel_D = criterion_pixelwise(fake_BD, real_BD)


        # Total loss
        loss_G = 0.5 * (loss_GAN + 0.1*loss_GAN_D + lambda_pixel * (loss_pixel + 0.1*loss_pixel_D) )

        #loss_G_list.append(loss_G.numpy())

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        pred_real_D = discriminator(real_BD, real_AD)
        loss_real_D = criterion_GAN(pred_real_D, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        pred_fake_D = discriminator(fake_BD.detach(), real_AD)
        loss_fake_D = criterion_GAN(pred_fake_D, fake)

        # Total loss
        loss_D = 0.25 * (loss_real + loss_fake + loss_real_D + loss_fake_D)
        
        #loss_D_list.append(loss_D.numpy())
        
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * 10000000000 + len(dataloader) * 100000 + i
        #batches_left = opt.n_epochs * len(dataloader) - batches_done
        #time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        #prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item()))

        # If at sample interval save image
        if epoch % opt.sample_interval == 0:
            sample_images(batches_done)


            
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (opt.datasave_name, epoch))
        torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (opt.datasave_name, epoch))





'''
x1 = range(opt.epoch, opt.n_epochs)
x2 = range(opt.epoch, opt.n_epochs)
y1 = loss_G_list
y2 = loss_D_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test loss_G vs. epoches')
plt.ylabel('Test loss_G')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss_D vs. epoches')
plt.ylabel('Test loss_D')
plt.show()
plt.savefig("loss_G and loss_Ds.png")
'''
