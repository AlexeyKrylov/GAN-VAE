import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
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
                init.constant_(m.bias.data, 0.0)
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
             gpu_ids=[], isize=0):
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
        net = CelebaGenerator(isize, output_nc, ngf, nz, norm_layer=norm_layer)
    elif netG == 'mnist_generator':
        net = MNISTGenerator(output_nc, ngf, nz, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], isize=0):
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
        net = CelebaDiscriminator(isize, input_nc, ndf, norm_layer=norm_layer)
    elif netD == "mnist_discriminator":
        net = MNISTDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_Encoder(input_nc, ngf, nz, net_encoder, norm='batch', init_type='normal', init_gain=0.02,
                   gpu_ids=[], isize=0):
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
        net = CelebaEncoder(isize, input_nc, ngf, nz, norm_layer=norm_layer)
    elif net_encoder == 'mnist_encoder':
        net = MNISTEncoder(input_nc, ngf, nz, norm_layer=norm_layer)
    elif net_encoder == 'rls_encoder':
        net = resnet18(num_classes=nz)
        # net = CelebaEncoder(isize, input_nc, ngf, nz, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net_encoder)
    return init_net(net, init_type, init_gain, gpu_ids), net.kl


def define_Decoder(output_nc, ndf, nz, net_decoder, norm='batch', init_type='normal', init_gain=0.02,
                   gpu_ids=[], isize=0):
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
        net = CelebaDecoder(isize, output_nc, ndf, nz, norm_layer=norm_layer)
    elif net_decoder == 'mnist_decoder':
        net = MNISTDecoder(output_nc, ndf, nz, norm_layer=norm_layer)
    elif net_decoder == 'rls_decoder':
        # net = RlsDecoder(isize, output_nc, ndf, nz, norm_layer=norm_layer)
        # net = CelebaDecoder(isize, output_nc, ndf, nz, norm_layer=norm_layer)
        net = ResNet18Dec(isize=isize, z_dim=nz, nc=output_nc)
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


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            block = block.to(device='cuda')
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class VAELoss(nn.Module):
    """Define different VAE objectives."""

    def __init__(self):
        """Initialize the VAELoss class."""
        super(VAELoss, self).__init__()
        loss = nn.L1Loss()
        perceptloss = VGGPerceptualLoss(resize=True)
        self.loss = lambda prediction, real, kl: loss(prediction, real) + kl + perceptloss(prediction, real)

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
    def __init__(self, isize, output_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaGenerator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf *= 2
            tisize *= 2

        model = []

        model += [
            nn.ConvTranspose2d(nz, cngf, 4, stride=1, padding=0, output_padding=0, bias=use_bias),
            norm_layer(cngf),
            nn.ReLU(True)
        ]

        csize = 4

        while csize < isize // 2:
            model += [
                nn.ConvTranspose2d(cngf, cngf // 2, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
                norm_layer(cngf // 2),
                nn.ReLU(True),
            ]
            cngf = cngf // 2
            csize *= 2

        model += [
            nn.ConvTranspose2d(cngf, output_nc, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
        ]

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class CelebaDiscriminator(nn.Module):
    def __init__(self, isize, input_nc, ndf, norm_layer=nn.BatchNorm2d):
        super(CelebaDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            # input is (nc) x 64 x 64
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        csize, cndf = isize // 2, ndf

        while csize > 4:
            model += [nn.Conv2d(cndf, cndf * 2, 4, 2, 1, bias=use_bias),
                      norm_layer(cndf * 2),
                      nn.LeakyReLU(0.2, inplace=True),
            ]
            cndf *= 2
            csize //= 2

        model += [
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(cndf, 1, 4, 1, 0, bias=use_bias),
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
                                      opt.init_gain, opt.gpu_ids, isize=opt.crop_size)
        self.decoder = define_Decoder(opt.output_nc, opt.ngf, opt.nz, opt.net_decoder, opt.norm, opt.init_type,
                                      opt.init_gain, opt.gpu_ids, isize=opt.crop_size)
        self.opt = opt

        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)


class Autoencoder(nn.Module):
    def __init__(self, opt):
        super(Autoencoder, self).__init__()
        self.encoder, _ = define_Encoder(opt.input_nc, opt.ngf, opt.nz, opt.net_encoder, opt.norm, opt.init_type,
                                      opt.init_gain, opt.gpu_ids, isize=opt.crop_size)
        self.decoder = define_Decoder(opt.output_nc, opt.ngf, opt.nz, opt.net_decoder, opt.norm, opt.init_type,
                                      opt.init_gain, opt.gpu_ids, isize=opt.crop_size)
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
        self.kl = ((sigma ** 2 + mu ** 2 - torch.log(sigma) - 1) / 2).sum()
        return z


class CelebaDecoder(nn.Module):
    def __init__(self, isize, output_nc, ndf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaDecoder, self).__init__()

        cndf, tisize = ndf // 2, 4
        while tisize != isize:
            cndf *= 2
            tisize *= 2

        use_bias = False
        model = []

        model += [nn.Unflatten(dim=1, unflattened_size=((nz), 1, 1))]

        model += [
            nn.ConvTranspose2d(nz, cndf, 4, stride=1, padding=0, output_padding=0, bias=use_bias),
            norm_layer(cndf),
            nn.ReLU(True)
        ]

        csize = 4

        while csize < isize // 2:
            model += [
                nn.ConvTranspose2d(cndf, cndf // 2, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
                norm_layer(cndf // 2),
                nn.ReLU(True),
            ]
            cndf = cndf // 2
            csize *= 2
        model += [
            nn.ConvTranspose2d(cndf, output_nc, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
        ]

        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if input.ndim != 2:
            input = torch.reshape(input, (input.shape[0], input.shape[1]))
        return self.model(input)


class CelebaEncoder(nn.Module):
    def __init__(self, isize, input_nc, ngf, nz, norm_layer=nn.BatchNorm2d):
        super(CelebaEncoder, self).__init__()

        use_bias = False

        model = [nn.Conv2d(input_nc, ngf, 4, stride=2, padding=1, bias=use_bias),
                 nn.ReLU(True)
                 ]

        csize, cngf = isize // 2, ngf
        while csize > 4:
            in_feat = cngf
            out_feat = cngf * 2
            model += [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1, bias=use_bias),
                     norm_layer(out_feat),
                     nn.ReLU(True)
            ]
            cngf *= 2
            csize //= 2
        model += [nn.Conv2d(cngf, cngf * 2, 4, stride=2, padding=0, bias=use_bias),
                  nn.ReLU(True)
                  ]
        self.model = nn.Sequential(*model)
        self.linear2 = nn.Linear((cngf * 2), nz)
        self.linear3 = nn.Linear((cngf * 2), nz)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, input):
        x = self.model(input)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class RlsDecoder(nn.Module):
    def __init__(self, isize, output_nc, ndf, nz, norm_layer=nn.BatchNorm2d):
        super(RlsDecoder, self).__init__()
        cndf, tisize = ndf // 2, 4
        while tisize != isize:
            cndf *= 2
            tisize *= 2

        use_bias = False
        model = []

        model += [nn.Unflatten(dim=1, unflattened_size=((nz), 1, 1))]

        model += [
            nn.ConvTranspose2d(nz, cndf, 4, stride=1, padding=0, output_padding=0, bias=use_bias),
            norm_layer(cndf),
            nn.ReLU(True)
        ]

        csize = 4

        while csize < isize // 2:
            model += [
                nn.ConvTranspose2d(cndf, cndf // 2, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
                norm_layer(cndf // 2),
                nn.ReLU(True),
            ]
            cndf = cndf // 2
            csize *= 2
        model += [
            nn.ConvTranspose2d(cndf, output_nc, 4, stride=2, padding=1, output_padding=0, bias=use_bias),
        ]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if input.ndim != 2:
            input = torch.reshape(input, (input.shape[0], input.shape[1]))
        return self.model(input)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=scale_factor, padding=1, output_padding=1)

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        #x = self.conv(x)
        x = self.convt(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop1 = nn.Dropout(0.1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.drop1(out)

        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class
        self.drop = nn.Dropout2d(0.1)
        self.drop1 = nn.Dropout2d(0.1)
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.drop(torch.relu(self.bn2(self.conv2(x))))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = self.drop1(torch.relu(out))
        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, donwsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = donwsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.linear2 = nn.Linear(512 * block.expansion, num_classes)
        self.linear3 = nn.Linear(512 * block.expansion, num_classes)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z

    def forward(self, x):
        return self._forward_impl(x)

class ResNet18Dec(nn.Module):

    def __init__(self, isize, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512
        self.isize = isize
        self.nc = nc
        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        if z.ndim != 2:
            z = torch.reshape(z, (z.shape[0], z.shape[1]))
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=self.isize // 16)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, self.isize, self.isize)
        return x

def _resnet(arch, block, layers, **kwargs):
    print(arch)
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext50_32x4d(progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnet50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet50_2(progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', "Bottleneck", [3, 4, 6, 3], **kwargs)