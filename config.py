import torchvision.transforms as transforms


class Config:

    def __init__(self):

        self.lr = 4e-3

        self.epoch_num = 100
        self.batch_size = 256
        self.shuffle = True

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224), scale=(0.08, 1.0),ratio=(0.75, 4. / 3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0),
            transforms.Normalize((123.68, 116.779, 103.939), (58.393, 57.12, 57.375)),
            transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.Normalize((123.68, 116.779, 103.939), (58.393, 57.12, 57.375)),
            transforms.ToTensor()
        ])

        self.test_every = 5
        self.save_checkpoint = True
        self.save_every = 10

        self.train_path = '/media/lizishi/本地磁盘/data2train/train'
        self.test_path = '/media/lizishi/本地磁盘/data2train/test'
        self.save_path = '/media/lizishi/本地磁盘/pattern-classification-checkpoint_3'
