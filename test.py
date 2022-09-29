from train import *
import matplotlib.pyplot as plt
from utils import *
import torchvision
import torchvision.transforms as transforms
import random

if __name__ == '__main__':
    from data import MNIST_generation

    # ray.init()

    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               download=False,
                                               transform=transforms.ToTensor())
    train_num = 20
    assert train_num <= len(train_dataset)
    idx = [i for i in range(20)]#random.sample(list(range(len(train_dataset))), train_num)
    train_dataset.data = train_dataset.data[idx]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)
    for i, (images, labels) in enumerate(train_loader):
        plt.imshow(images[0].squeeze(0))
        plt.show()
        print(labels[0])
        pass

