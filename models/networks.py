import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


##############################################################################
# Helper Functions
##############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='batch'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer           -- the optimizer of the network
        opt (option class)  -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                               opt.lr_policy is the name of learning rate policy: step | plateau | cosine | None

    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytoch.org/docs/stable/optim.html for more details.
    """

    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == None:
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original Celeba and MNIST GANs. But xavier and kaiming might work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)       -- the network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)        -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)  -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(output_nc, ngf, nz, netG, norm='batch', init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int)      -- the number of channels in input images
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        netG (str)          -- the architecture's name: celeba_generator | mnist_generator
        norm (str)          -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)     -- the name of our initialization method.
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)  -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'celeba_generator':
        net = CelebaGenerator(output_nc, ngf, nz, norm_layer=norm_layer)
    elif netG == 'mnist_generator':
        net = MNISTGenerator(output_nc, ngf, nz, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input _nc (int)     -- the number of channels in input images
        ndf (int)           -- the number of filters in the first conv layer
        netD (str)          -- the architecture's name: basic | n_layers | pixel | custom_discriminator
        n_layers_D (int)    -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)          -- the type of normalization layers used in the network.
        init_type (str)     -- the name of the initialization method.
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)  -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'celeba_discriminator':
        net = CelebaDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "mnist_discriminator":
        net = MNISTDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_Encoder(input_nc, ngf, nz, net_encoder, norm='batch', init_type='normal', init_gain=0.02,
                   gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int)      -- the number of channels in input images
        ngf (int)           -- the number of filters in the last conv layer
        net_encoder (str)          -- the architecture's name: celeba_encoder | mnist_encoder
        norm (str)          -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)     -- the name of our initialization method.
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)  -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if net_encoder == 'celeba_encoder':
        net = CelebaEncoder(input_nc, ngf, nz, norm_layer=norm_layer)
    elif net_encoder == 'mnist_encoder':
        net = MNISTEncoder(input_nc, ngf, nz, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net_encoder)
    return init_net(net, init_type, init_gain, gpu_ids), net.kl


def define_Decoder(output_nc, ndf, nz, net_decoder, norm='batch', init_type='normal', init_gain=0.02,
                   gpu_ids=[]):
    """Create a generator

    Parameters:
        output_nc (int)      -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        net_decoder (str)          -- the architecture's name: celeba_decoder | mnist_decoder
        norm (str)          -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)     -- the name of our initialization method.
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)  -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if net_decoder == 'celeba_decoder':
        net = CelebaDecoder(output_nc, ndf, nz, norm_layer=norm_layer)
    elif net_decoder == 'mnist_decoder':
        net = MNISTDecoder(output_nc, ndf, nz, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net_decoder)
    return init_net(net, init_type, init_gain, gpu_ids)


#########################################################################################################
# Classes
#########################################################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str)              -- the type of GAN objective. It currently supports vanilla, Isgan, and wganp.
            target_real_label (bool)    -- label for a real image
            target_fake_label (bool)    -- label of a fake image
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'dcgan':
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            predictions (tensor)        -- typically the predictions from discriminator
            target_is_real (bool)       -- if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor)     -- typically the prediction output from a discriminator
            target_is_real (bool)   -- if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class VAELoss(nn.Module):
    """Define different VAE objectives."""

    def __init__(self):
        """Initialize the VAELoss class."""
        super(VAELoss, self).__init__()
        self.loss = lambda prediction, real, kl: ((prediction - real)**2).sum() + kl

    def __call__(self, prediction, real, kl):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor)     -- typically the prediction output from a discriminator
            real (tensor)   -- real image

        Returns:
            the calculated loss.
        """
        loss = self.loss(prediction, real, kl)
        return loss


class CelebaGenerator(nn.Module):
    def __init__(self, output_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaGenerator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1, bias=use_bias),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CelebaDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
        super(CelebaDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            # input is (nc) x 64 x 64
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=use_bias),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, inplace=use_bias),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_bias),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class MNISTGenerator(nn.Module):
    def __init__(self, output_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(MNISTGenerator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) * 4 * 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) * 8 * 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            # state size. (ngf) * 16 * 16
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 3, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class MNISTDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
        super(MNISTDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, 4, 2, 3, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) * 16 * 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) * 8 * 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) * 4 * 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=use_bias),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class VariationalAutoencoder(nn.Module):
    def __init__(self, opt):
        super(VariationalAutoencoder, self).__init__()
        self.encoder, self.kl = define_Encoder(opt.input_nc, opt.ngf, opt.nz, opt.net_encoder, opt.norm, opt.init_type,
                                      opt.init_gain, opt.gpu_ids)
        self.decoder = define_Decoder(opt.output_nc, opt.ngf, opt.nz, opt.net_decoder, opt.norm, opt.init_type,
                                      opt.init_gain, opt.gpu_ids)
        self.opt = opt

        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)


class MNISTDecoder(nn.Module):
    def __init__(self, output_nc, ndf, nz, norm_layer=nn.BatchNorm2d):
        super(MNISTDecoder, self).__init__()

        use_bias = True

        model = [
            nn.Linear(nz, (ndf * 16)),
            nn.ReLU(True),
            nn.Linear((ndf * 16), 3 * 3 * (ndf * 4)),
            nn.ReLU(True)
        ]

        model += [nn.Unflatten(dim=1, unflattened_size=((ndf * 4), 3, 3))]

        model += [
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 3, stride=2, output_padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf, output_nc, 3, stride=2, padding=1, output_padding=1, bias=use_bias)  # ngf = 8
        ]

        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if input.ndim != 2:
            input = torch.reshape(input, (input.shape[0], input.shape[1]))
        return self.model(input)


class MNISTEncoder(nn.Module):
    def __init__(self, input_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(MNISTEncoder, self).__init__()

        use_bias = True

        self.conv1 = nn.Conv2d(input_nc, ngf, 3, stride=2, padding=1, bias=use_bias)
        self.batch1 = norm_layer(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=use_bias)
        self.batch2 = norm_layer(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=0, bias=use_bias)
        self.linear1 = nn.Linear(3 * 3 * (ngf * 4), (ngf * 16))
        self.linear2 = nn.Linear((ngf * 16), nz)
        self.linear3 = nn.Linear((ngf * 16), nz)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, input):
        x = F.relu(self.batch1(self.conv1(input)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class CelebaDecoder(nn.Module):
    def __init__(self, output_nc, ndf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaDecoder, self).__init__()

        use_bias = True

        model = [
            nn.Linear(nz, (ndf * 32)),
            nn.ReLU(True),
            nn.Linear((ndf * 32), 3 * 3 * (ndf * 8)),
            nn.ReLU(True)
        ]

        model += [nn.Unflatten(dim=1, unflattened_size=((ndf * 8), 3, 3))]

        model += [
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 3, stride=2, padding=0, output_padding=1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf, output_nc, 4, stride=2, padding=1, output_padding=0, bias=use_bias)
        ]

        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if input.ndim != 2:
            input = torch.reshape(input, (input.shape[0], input.shape[1]))
        return self.model(input)


class CelebaEncoder(nn.Module):
    def __init__(self, input_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaEncoder, self).__init__()

        use_bias = True

        self.conv0 = nn.Conv2d(input_nc, ngf, 5, stride=2, padding=0, bias=use_bias)
        self.batch0 = norm_layer(ngf)
        self.conv1 = nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=0, bias=use_bias)
        self.batch1 = norm_layer(ngf * 2)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=use_bias)
        self.batch2 = norm_layer(ngf * 4)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, 3, stride=2, padding=0, bias=use_bias)
        self.linear1 = nn.Linear(3 * 3 * (ngf * 8), (ngf * 32))
        self.linear2 = nn.Linear((ngf * 32), nz)
        self.linear3 = nn.Linear((ngf * 32), nz)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, input):
        x = F.relu(self.batch0(self.conv0(input)))
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z
