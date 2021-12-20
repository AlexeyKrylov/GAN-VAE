import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class DCGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generator and discriminator)
        self.netG = networks.define_G(opt.output_nc, opt.ngf, opt.nz, opt.netG, opt.norm, opt.init_type, opt.init_gain,
                                      self.gpu_ids, isize=opt.crop_size)

        if self.isTrain:  # define discriminator
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.norm, opt.init_type, opt.init_gain,
                                          self.gpu_ids, isize=opt.crop_size)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.k_D = opt.k_D
        self.nz = opt.nz
        self.global_loss_D = 0
        self.global_loss_G = 0
        self.fixed_noise = torch.randn(9, opt.nz, 1, 1,
                                       device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def compute_visuals(self, epoch):
        """Calculate additional output images for visdom and HTML visualization"""

        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
        plt.imsave(f"{self.opt.image_pred_dir}\image{epoch}.png",
                   np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)).cpu().numpy())

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input[0] (dict): include the data itself and its metadata information.
        """
        self.real = input[0].to(self.device)
        self.b_size = self.real.size(0)

    def forward(self):
        # Generate batch of latent vectors
        noise = torch.randn(self.b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        self.fake = self.netG(noise)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_real = self.netD(self.real).view(-1)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD(self.fake.detach()).view(-1)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.global_loss_D += loss_D
        loss_D.backward()
        return loss_D

    def backward_G(self):
        """Calculate the loss for generator G"""
        self.loss_G = self.criterionGAN(self.netD(self.fake).view(-1), True)
        self.loss_G.backward()
        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_requires_grad(self.netD, True)
        for step in range(self.k_D):
            self.optimizer_D.zero_grad()
            self.forward()
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.global_loss_D /= self.k_D
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.global_loss_G = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()
        ret_loss_D = self.global_loss_D
        ret_loss_G = self.global_loss_G
        self.global_loss_D = 0
        self.global_loss_G = 0
        return ret_loss_G, ret_loss_D
