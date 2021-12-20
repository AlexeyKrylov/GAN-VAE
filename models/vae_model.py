import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class VAEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.iii = 1
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['VAE']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['VAE']

        # define network
        self.netVAE = networks.VariationalAutoencoder(opt)

        if self.isTrain:
            # define loss function
            self.criterionVAE = networks.VAELoss().to(self.device)  # define VAE loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netVAE.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)

        self.nz = opt.nz
        self.global_loss = 0

        Norm = torch.distributions.Normal(0, 1)

        mu = torch.randn(8, opt.nz, 1, 1,
                                       device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        sigma = torch.randn(8, opt.nz, 1, 1,
                                       device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


        self.fixed_noise = mu + sigma * Norm.sample(mu.shape).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def compute_visuals(self, epoch):
        """Calculate additional output images for visdom and HTML visualization"""

        with torch.no_grad():
            fake = self.netVAE.decoder(self.fixed_noise).detach().cpu()
        plt.imsave(f"{self.opt.image_pred_dir}\image{epoch}.png",
                   np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)).cpu().numpy())
        plt.imsave(f"{self.opt.image_pred_dir}\image_fake{self.iii}.png",
                   np.transpose(vutils.make_grid(self.fake.detach().cpu(), padding=2, normalize=True), (1, 2, 0)).cpu().numpy())
        plt.imsave(f"{self.opt.image_pred_dir}\image_real{self.iii}.png",
                   np.transpose(vutils.make_grid(self.real.detach().cpu(), padding=2, normalize=True),
                               (1, 2, 0)).cpu().numpy())
        self.iii += 1

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input[0] (dict): include the data itself and its metadata information.
        """
        self.real = input[0].to(self.device)
        self.b_size = self.real.size(0)

    def forward(self):
        # Generate fake image batch
        self.fake = self.netVAE(self.real)
        # plt.imsave(f"{self.opt.image_pred_dir}\image_fake{self.iii}.png",
        #            np.transpose(vutils.make_grid(self.fake.detach().cpu(), padding=2, normalize=True), (1, 2, 0)).cpu().numpy())
        # plt.imsave(f"{self.opt.image_pred_dir}\image_real{self.iii}.png",
        #           np.transpose(vutils.make_grid(self.real.detach().cpu(), padding=2, normalize=True),
        #                       (1, 2, 0)).cpu().numpy())
        # self.iii += 1

    def backward(self):
        """Calculate the loss for generator G"""
        self.loss = self.criterionVAE(self.fake, self.real, self.netVAE.kl)
        self.loss.backward()
        return self.loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.set_requires_grad(self.netVAE, True)
        self.optimizer.zero_grad()
        self.forward()
        self.global_loss = self.backward()
        self.optimizer.step()
        ret_loss = self.global_loss
        self.global_loss = 0
        return ret_loss
