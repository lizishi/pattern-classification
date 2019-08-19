import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from tqdm import tqdm

from config import Config
from dataset import ImageFolder

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'

model = torchvision.models.resnet101(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(in_features=2048, out_features=2)
model = model.to(device)
sigmoid = nn.Sigmoid()


def run(config):
    train_dataset = ImageFolder(config.train_path, transform=config.transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=config.shuffle)

    test_dataset = ImageFolder(config.test_path, transform=config.transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config.batch_size,
                                                  shuffle=config.shuffle)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    loss_function = nn.BCELoss()

    for epoch in range(config.epoch_num):
        if epoch and epoch % 20 == 0:
            config.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr
        model.train()
        sum_loss = 0
        for index, data in tqdm(enumerate(train_dataloader)):
            img = data[0].float().to(device)
            label = data[1].float().to(device)

            optimizer.zero_grad()
            label_ = model(img)
            label_ = sigmoid(label_)
            loss = loss_function(label_, label)

            loss.backward()
            optimizer.step()

            sum_loss += loss.sum().item()

        print('epoch: %d, loss: %f' % (epoch + 1, sum_loss))

        if (epoch + 1) % config.test_every == 0:
            model.eval()
            with torch.no_grad():
                acc1 = 0
                acc2 = 0
                for index, data in tqdm(enumerate(test_dataloader)):
                    img = data[0].float().to(device)
                    label = data[1].float().to(device)

                    label_ = model(img)
                    label_ = sigmoid(label_).round()

                    equal = (label == label_).cpu().numpy()
                    for vector in equal:
                        if vector[0]:
                            acc1 += 1
                        if vector[1]:
                            acc2 += 1

            acc1 /= len(test_dataset)
            acc2 /= len(test_dataset)
            print('first category acc: %f, second category acc: %f' %
                  (acc1, acc2))

        if config.save_checkpoint and (epoch + 1) % config.save_every == 0:
            if not os.path.exists(config.save_path):
                os.makedirs(config.save_path)
            torch.save(model, os.path.join(config.save_path, 'resnet101_%d.pth' % epoch))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        conf = Config()
    else:
        conf = getattr(Config, sys.argv[1])
    run(conf)
