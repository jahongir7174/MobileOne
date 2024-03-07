import os
from os.path import *

from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform

        self.samples, self.cls_names = self.load_label(data_dir)

    def __getitem__(self, index):
        filename, label = self.samples[index]

        image = self.load_image(filename)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_label(data_dir):
        images = []
        labels = []

        cls_names = [x for x in os.listdir(data_dir)]
        cls_names.sort()

        cls_to_idx = {cls_name: i for i, cls_name in enumerate(cls_names)}

        for root, dirs, filenames in os.walk(data_dir, topdown=False, followlinks=True):

            for filename in filenames:
                base, extension = splitext(filename)
                if extension.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(join(root, filename))
                    labels.append(basename(root))

        return [(i, cls_to_idx[j]) for i, j in zip(images, labels) if j in cls_to_idx], cls_names
