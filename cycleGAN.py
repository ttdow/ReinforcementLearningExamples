import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import glob
import time

import torchvision.transforms as transforms
from torchvision.utils import make_grid

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)
    
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, height*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]

        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height//2**4, width//2**4)

        # Returns downsampling layers of each discriminator block
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]

            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),     # Left and top padding
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        
        if self.mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(root + '/  _jpg') + '/*.*')[:250])
            self.files_B = sorted(glob.glob(os.path.join(root + '/photo_jpg') + '/*.*')[:250])
        elif self.mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(root + '/monet_jpg') + '/*.*')[250:])
            self.files_B = sorted(glob.glob(os.path.join(root + '/photo_jpg') + '/*.*')[250:301])

        # Convert image to RGB
    def to_rgb(self, image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)

        return rgb_image

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != 'RGB':
            image_A = self.to_rgb(image_A)

        if image_B.mode != 'RGB':
            image_B = self.to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {'A':item_A, 'B':item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
# Learning rate scheduler
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert(n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def main():
    # Input file path
    root = './data/'

    # Input image data
    img_height = 256
    img_width = 256
    n_channels = 3

    # Hyperparameters
    n_epochs = 5
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    decay_epoch = 3 # Epoch from which to start lr decay

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Initialize generators and discriminators
    input_shape = (n_channels, img_height, img_width)
    n_residual_blocks = 9 # Suggested default, number of residual blocks in generator

    #       ----------------------------------- D_A -------------------------------------
    #       |                                                                           |
    # Real image in domain A -> G_AB -> Fake image in domain B -> G_BA -> Reconstructed image in domain A
    #                                           |
    #                   Real or fake?  <- D_B  <+
    #                                           |
    #                                   Real image in domain B

    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    criterion_identity.to(device)

    # Weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # Reset Conv2d's weight (tensor) with Gaussian dist

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # Reset Conv2d's bias (tensor) with 0
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # Reset BatchNorm2d's weight (tensor) with Gaussian dist
            torch.nn.init.constant_(m.bias.data, 0.0) # Reset BatchNorm2d's bias (tensor) with 0

    # Apply weight initialization
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Configure optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)

    # Image transformation setting
    transforms_ = [
        transforms.Resize(int(img_height*1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = DataLoader(
        ImageDataset(root, transforms_=transforms_, unaligned=True),
        batch_size = 1,
        shuffle = True,
        num_workers = 1
    )

    val_dataloader = DataLoader(
        ImageDataset(root, transforms_=transforms_, unaligned=True, mode='test'),
        batch_size = 5,
        shuffle = True,
        num_workers = 1
    )

    Tensor = torch.Tensor

    # Show a generated sample of images from the test set
    def sample_images():
        imgs = next(iter(val_dataloader))
        
        G_AB.eval() # Sets the module in evaluation mode?
        G_BA.eval()

        real_A = imgs['A'].type(Tensor) # Monet
        fake_B = G_AB(real_A).detach()
        real_B = imgs['B'].type(Tensor) # Photo
        fake_A = G_BA(real_B).detach()

        # Arrange images along the x-axis
        real_A = make_grid(real_A, nrows=5, normalize=True)
        fake_A = make_grid(fake_A, nrows=5, normalize=True)
        real_B = make_grid(real_B, nrows=5, normalize=True)
        fake_B = make_grid(fake_B, nrows=5, normalize=True)

        # Arrange images along the y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        plt.imshow(image_grid.cpu().permute(1,2,0))
        plt.title('Real A vs Fake B | Real B vs Fake A')
        plt.axis('off')
        plt.show()

    temp_imgs = next(iter(val_dataloader))

    G_AB.eval()
    G_BA.eval()

    temp_real_A = temp_imgs['A'].type(Tensor) # Monet
    temp_fake_B = G_AB(temp_real_A).detach()
    temp_real_B = temp_imgs['B'].type(Tensor) # Photo
    temp_fake_A = G_BA(temp_real_B).detach()

    temp_real_A = make_grid(temp_real_A, nrow=5, normalize=True)
    temp_real_B = make_grid(temp_real_B, nrow=5, normalize=True)
    temp_fake_A = make_grid(temp_fake_A, nrow=5, normalize=True)
    temp_fake_B = make_grid(temp_fake_B, nrow=5, normalize=True)

    temp_image_grid = torch.cat((temp_real_A, temp_fake_A, temp_real_B, temp_fake_B), 1)
    temp_image_grid.cpu().permute(1,2,0).shape

    #plt.imshow(temp_image_grid.cpu().permute(1,2,0))
    #plt.title('Real A | Fake B | Real B | Fake A')
    #plt.axis('off')
    #plt.show()

    old_time = time.time()

    # Training
    print("Beginning training...")
    for epoch in range(0, n_epochs):
        for i, batch in enumerate(dataloader):

            print("Step: ", i)

            # Set model input
            real_A = batch['A'].type(Tensor)
            real_B = batch['B'].type(Tensor)

            # Adversarial ground truths
            valid = Tensor(np.ones((real_A.size(0), *D_A.output_shape))) # Requires grad = False (default)
            fake = Tensor(np.zeros((real_A.size(0), *D_A.output_shape))) # Requires grad = False (default)

            # ---------- Train generators ----------
            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad() # Integrated optimizer(G_AB, G_BA)

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A) # If you put A into a generator that creates A with B
            loss_id_B = criterion_identity(G_AB(real_B), real_B) # Then, of course, A must come out as it is

            # Taking this into consideration, add an identity loss that simply compares A and A (or B and B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A) # fake_B is a fake photo generated by real Monet painting
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) # Tricking the fake B into real B

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid) # Tricking the fake A into real A

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B) # recov_A is a fake Monet painting generated by a fake photo
            loss_cycle_A = criterion_cycle(recov_A, real_A) # Reduces the difference between the recovered image and the real image

            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + (10.0 * loss_cycle) + (5.0 * loss_identity) # Multiply suggested weight

            loss_G.backward()
            optimizer_G.step()

            # ---------- Train discriminator A ----------
            optimizer_D_A.zero_grad()

            loss_real_A = criterion_GAN(D_A(real_A), valid) # Train to discriminate real images as real

            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake) # Train to discriminate fake images as fake

            loss_D_A = (loss_real_A + loss_fake_A) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------- Train discriminator B ----------
            optimizer_D_B.zero_grad()

            loss_real_B = criterion_GAN(D_B(real_B), valid) # Train to discriminate real images as real

            loss_fake_B = criterion_GAN(D_B(fake_A).detach(), fake) # Train to discriminate fake images as fake

            loss_D_B = (loss_real_B + loss_fake_B) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            # Total loss
            loss_D = (loss_D_A + loss_D_B) / 2

            # Show training progress
            if (i+1) % 50 == 0:
                sample_images()            
                print('[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G Loss: %f - (adv: %f, cycle: %f, identity: %f)]'
                      %(epoch+1, 
                        n_epochs, 
                        i+1, 
                        len(dataloader), 
                        loss_D.item(), 
                        loss_G.item(), 
                        loss_GAN.item(), 
                        loss_cycle.item(), 
                        loss_identity.item())
                      )

            new_time = time.time()
            print("Time: ", new_time - old_time) # Seconds
            old_time = new_time

if __name__ == "__main__":
    main()