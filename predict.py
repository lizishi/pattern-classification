import os
import numpy as np
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# model
model_path = '/media/lizishi/本地磁盘/pattern-classification-checkpoint_2/model_99.pth'
model = torch.load(model_path)

save_path_00 = '/media/lizishi/本地磁盘/data_by_model_2/00'
save_path_01 = '/media/lizishi/本地磁盘/data_by_model_2/01'
save_path_10 = '/media/lizishi/本地磁盘/data_by_model_2/10'
save_path_11 = '/media/lizishi/本地磁盘/data_by_model_2/11'
if not os.path.exists(save_path_00):
    os.makedirs(save_path_00)
if not os.path.exists(save_path_01):
    os.makedirs(save_path_01)
if not os.path.exists(save_path_10):
    os.makedirs(save_path_10)
if not os.path.exists(save_path_11):
    os.makedirs(save_path_11)
folder_label = {'00': np.array((0, 0)), '01': np.array((0, 1)),
                '10': np.array((1, 0)), '11': np.array((1, 1))}


class ImageFolder(data.Dataset):

    def __init__(self, transform=None):

        self.transform = transform
        self.main_folder = r'/media/lizishi/本地磁盘/data2train/test'
        self.folder_name = ['00', '01', '10', '11']
        self.file_list = []
        self.label_list = []
        for folder in self.folder_name:
            folder_path = os.path.join(self.main_folder, folder)
            for file in os.listdir(folder_path):
                self.file_list.append(os.path.join(folder_path, file))
                self.label_list.append(folder_label[folder])

    def __getitem__(self, index):
        path = self.file_list[index]
        label = self.label_list[index]
        return path, label

    def __len__(self):
        return len(self.file_list)


def main():
    image_size = 299
    crop_size = 299
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 1
    dataset = ImageFolder(transform=transform)
    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=False)
    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        for idx, path in tqdm(enumerate(data_loader)):
            # print(path[0])
            img_origin = Image.open(path[0][0]).convert('RGB')
            img = transform(img_origin)
            y = path[1].float().to(device)

            img_in = img.float().unsqueeze(0).to(device)
            y_ = model(img_in)
            y_ = sigmoid(y_).round()

            if (y != y_).any():
                ynp = y[0].cpu().numpy().astype(np.int)
                y_np = y_[0].cpu().numpy().astype(np.int)
                file_name = str(ynp[0]) + str(ynp[1]) + '_' + str(y_np[0]) + str(y_np[1]) + '.png'
                if y_[0][0] == 0 and y_[0][1] == 0:
                    img_path = os.path.join(save_path_00, file_name)
                elif y_[0][0] == 0 and y_[0][1] == 1:
                    img_path = os.path.join(save_path_01, file_name)
                elif y_[0][0] == 1 and y_[0][1] == 0:
                    img_path = os.path.join(save_path_10, file_name)
                else:
                    img_path = os.path.join(save_path_11, file_name)

                img_origin.save(img_path)


if __name__ == '__main__':
    main()
