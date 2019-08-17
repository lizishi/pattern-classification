import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

folder_list = ['00', '01', '10', '11']
folder_label = {'00': np.array((0, 0)), '01': np.array((0, 1)),
                '10': np.array((1, 0)), '11': np.array((1, 1))}


class ImageFolder(data.Dataset):

    def __init__(self, path, transform=None):

        self.img_list = []
        self.label_list = []

        for folder in folder_list:
            folder_path = os.path.join(path, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                self.img_list.append(file_path)
                self.label_list.append(folder_label[folder])

        self.transform = transform

    def __getitem__(self, index):
        path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_list)
