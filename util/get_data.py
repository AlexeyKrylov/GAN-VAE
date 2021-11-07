from __future__ import print_function
import os
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename
import gzip


class GetData(object):
    """A Python script for downloading Celeba or MNIST datasets.

    Parameters:
        technique (str) -- One of: 'celeba' or 'mnist'.
        verbose (bool)  -- If True, print additional information.

    Examples:
        >>> from util.get_data import GetData
        >>> gd = GetData(technique='celeba')
        >>> new_data_path = gd.get(save_path='./datasets') # options will be displayed.

    Alternatively, You can use bash scripts: 'scripts/download_celeba_model.sh'
    and 'scripts/download_mnist_model.sh'.
    """

    def __init__(self, technique='celeba', verbose=True):
        url_dict = {
            'celeba': 'https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ',
            'mnist': 'http://yann.lecun.com/exdb/mnist/'
        }

        self.url = url_dict.get(technique.lower())
        self._verbose = verbose

    def _print(self, text):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, 'lxml')
        options = [h.text for h in soup.find_all('a', href=True)
                   if h.text.endswith(('.zip', '.gz'))]
        return options

    def _present_options(self):
        r = requests.get(self.url)
        options = self._get_options(r)
        print('Options:\n')
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input("\nPlease enter the number of the "
                       "dataset above you wish to download:")
        return options[int(choice)]

    def _download_data(self, dataset_url, save_path):
        if not isdir(save_path):
            os.makedirs(save_path)

        base = basename(dataset_url)
        temp_save_path = join(save_path, base)

        with open(temp_save_path, "wb") as f:
            r = requests.get(dataset_url)
            f.write(r.content)

        if base.endswith('.gz'):
            obj = gzip.open(temp_save_path, 'r')
        elif base.endswith('.zip'):
            obj = ZipFile(temp_save_path, 'r')
        else:
            raise ValueError("Unknown File Type: {0}.".format(base))

        self._print("Unpacking Data...")
        obj.extractall()
        obj.close()
        os.remove(temp_save_path)

    def get(self, save_path, dataset=None):
        """

        Download a dataset.

        Parameters:
            save_path (str) -- A directory to save the data to.
            dataset (str)   -- (optional). A specific dataset to download.
            Note: this must include the file extension.
            If None, options will be presented for you
            to choose from.

        Returns:
            save_path_full (str) -- the absolute path to the downloaded data.

        """
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset

        save_path_full = join(save_path, selected_dataset.split('.')[0])

        if isdir(save_path_full):
            warn("\n'{0}' already exists. Voiding Download.".format(
                save_path_full))
        else:
            self._print('Downloading Data...')
            url = "{0}/{1}".format(self.url, selected_dataset)
            self._download_data(url, save_path_full)

        return abspath(save_path_full)
