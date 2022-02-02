from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console and save in neptune.ai')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=0,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0.00005, help='initial weight decay for adam')
        parser.add_argument('--gan_mode', type=str, default='dcgan',
                            help='the type of GAN objective. [dcgan | vae | gpgan].')
        parser.add_argument('--lr_policy', type=str, default=None,
                            help='learning rate policy. [step | plateau | cosine | None]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--k_D', type=int, default=1, help='the number of discriminator\'s iterations')
        parser.add_argument('--image_pred_dir', type=str, default="imagedir",
                            help='the directory for saving pred images on the fixed noise')

        self.isTrain = True
        return parser
