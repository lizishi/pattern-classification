import os

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
model = torch.load(r'F:\pattern-classification-checkpoint\model_99.pth')

save_path_00= r'F:\data_by_model_2\00'
save_path_01 = r'F:\data_by_model_2\01'
save_path_10 = r'F:\data_by_model_2\10'
save_path_11 = r'F:\data_by_model_2\11'
if not os.path.exists(save_path_00):
    os.makedirs(save_path_00)
if not os.path.exists(save_path_01):
    os.makedirs(save_path_01)
if not os.path.exists(save_path_10):
    os.makedirs(save_path_10)
if not os.path.exists(save_path_11):
    os.makedirs(save_path_11)


class ImageFolder(data.Dataset):

    def __init__(self):

        self.main_folder = r'F:\pattern_data'
        self.folder_name = ['Animal Print', 'Burnout', 'Camouflage',
                            'Checked', 'Colour gradient', 'Colourful',
                            'Floral', 'Herringbone', 'Marl', 'nan',
                            'Paisley', 'Photo print', 'Pinstriped',
                            'Plain', 'Polka dot', 'Print', 'Striped']
        self.file_list = []
        for folder in self.folder_name:
            folder_path = os.path.join(self.main_folder, folder)
            for file in os.listdir(folder_path):
                self.file_list.append(os.path.join(folder_path, file))

    def __getitem__(self, index):
        path = self.file_list[index]
        return path

    def __len__(self):
        return len(self.file_list)


def main():
    image_size = 400
    crop_size = 299
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 1
    dataset = ImageFolder()
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        for idx, path in tqdm(enumerate(data_loader)):
            img_origin = Image.open(path[0]).convert('RGB')
            img = transform(img_origin)

            img_in = img.float().unsqueeze(0).to(device)
            y_ = model(img_in)
            y_ = sigmoid(y_).round()

            if y_[0] == 0 and y_[1] == 0:
                img_path = os.path.join(save_path_00, str(idx) + '.png')
            elif y_[0] == 0 and y_[1] == 1:
                img_path = os.path.join(save_path_01, str(idx) + '.png')
            elif y_[0] == 1 and y_[1] == 0:
                img_path = os.path.join(save_path_10, str(idx) + '.png')
            else:
                img_path = os.path.join(save_path_11, str(idx) + '.png')

            img_origin.save(img_path)


if __name__ == '__main__':
    main()
