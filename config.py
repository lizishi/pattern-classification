import torchvision.transforms as transforms


class Config:

    def __init__(self):
        self.resize_size = 400
        self.crop_size = 299

        self.lr = 1e-3

        self.epoch_num = 200
        self.batch_size = 64
        self.shuffle = True

        self.transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.test_every = 5
        self.save_checkpoint = True
        self.save_every = 10

        self.train_path = r'F:\data2train\train'
        self.test_path = r'F:\data2train\test'
        self.save_path = r'F:\pattern-classification-checkpoint_1'
