from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import os


class RlsDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.len_ = opt.max_dataset_size
        self.dataroot = opt.dataroot
        self.transform = get_transform(opt)
        print(self.transform)
        self.listdir = os.listdir(path=opt.dataroot)
        print(self.opt.dataroot)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        new_index = np.random.randint(0, 3, 1)[0]
        img_path = os.path.join(self.dataroot, self.listdir[new_index])
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        """Return the total number of images."""
        return self.len_