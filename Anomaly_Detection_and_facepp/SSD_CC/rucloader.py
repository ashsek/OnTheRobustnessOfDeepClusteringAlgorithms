from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import os


class CIFAR10RUC(datasets.CIFAR10):
    def __init__(self, root, transform, target_transform=None,train=True, download = False):
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
#         self.transform2 = transform2
#         self.transform3 = transform3
#         self.transform4 = transform4

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def __getitem__(self, index) :
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
#             img2 = self.transform2(img)
#             img3 = self.transform3(img)

        return img1, target, index
