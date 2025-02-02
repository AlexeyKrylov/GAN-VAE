import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class GPGANModel(BaseModel):
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
        self.loss_names = ['AE']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['AE']

        # define network
        self.netAE = networks.Autoencoder(opt)

        if self.isTrain:
            # define loss function
            self.criterionAE = networks.VAELoss().to(self.device)  # define VAE loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netAE.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)

        self.nz = opt.nz
        self.global_loss = 0
        self.fixed_noise = torch.randn(64, opt.nz, 1, 1,
                                       device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def compute_visuals(self, epoch):
        """Calculate additional output images for visdom and HTML visualization"""

        with torch.no_grad():
            fake = self.netAE.decoder(self.fixed_noise).detach().cpu()
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
        # Generate fake image batch
        self.fake = self.netAE(self.real)

    def backward(self):
        """Calculate the loss for generator G"""
        self.loss = self.criterionAE(self.fake, self.real, 0)
        self.loss.backward()
        return self.loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_requires_grad(self.netAE, True)
        self.optimizer.zero_grad()
        self.forward()
        self.global_loss = self.backward()
        self.optimizer.step()
        ret_loss = self.global_loss
        self.global_loss = 0
        return ret_loss