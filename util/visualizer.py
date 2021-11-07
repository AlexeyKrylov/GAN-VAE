import os
import time
from . import util
import neptune.new as neptune
from neptune.new.types import File


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information."""

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt  # cache the option

        self.run = neptune.init(
            project="alexeykrylov/GANs",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MDMyMGY5Yy1jNGVmLTQxYmItOGU5NC0xNmU3ZTlkYTBiNDEifQ=="
        )
        self.run["parameters"] = vars(opt)
        util.mkdir(self.opt.image_pred_dir)
        # create a logging file to store training losses

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, losses):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '(epoch: %d) ' % (epoch)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def upload_current_visuals(self, epoch):
        self.run[self.opt.image_pred_dir].log(File(f"{self.opt.image_pred_dir}\image{epoch}.png"))

    def upload_current_losses(self, losses):
        for k, v in losses.items():
            self.run[f'train/epoch/{k}'].log(v)
