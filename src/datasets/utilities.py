import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from .DownsampledImageNet import ImageNet16
from sklearn.model_selection import StratifiedKFold

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'imagenet-1k-s': 1000,
                 'imagenet-1k': 1000,
                 'ImageNet16' : 1000,
                 'ImageNet16-120': 120,
                 'ImageNet16-150': 150,
                 'ImageNet16-200': 200}

class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())


def get_datasets(name, root, input_size, cutout=-1):
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        input_size = input_size[1:]
    assert input_size[1] == input_size[2]

    if name == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std  = [0.24703233, 0.24348505, 0.26158768]
    elif name == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std  = [0.2673, 0.2564, 0.2762]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [0.481098, 0.45749, 0.407882]
        std  = [0.247922, 0.240235, 0.255255]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('ImageNet16'):
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        if cutout > 0 : lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif name.startswith('imagenet-1k'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k':
            xlists    = []
            xlists.append(transforms.Resize((input_size[1], input_size[1]), interpolation=2))
            xlists.append(transforms.RandomCrop(input_size[1], padding=0))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(input_size[1], scale=(0.2, 1.0))]
            xlists = []
        else: raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        xlists.append(RandChannel(input_size[0]))
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose([transforms.Resize(input_size[1]), transforms.CenterCrop(input_size[1]), transforms.ToTensor(), normalize])
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    elif name == 'ImageNet16':
        root = osp.join(root, 'ImageNet16')
        train_data = ImageNet16(root, True , train_transform)
        test_data  = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        root = osp.join(root, 'ImageNet16')
        train_data = ImageNet16(root, True , train_transform, 120)
        test_data  = ImageNet16(root, False, test_transform , 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        root = osp.join(root, 'ImageNet16')
        train_data = ImageNet16(root, True , train_transform, 150)
        test_data  = ImageNet16(root, False, test_transform , 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        root = osp.join(root, 'ImageNet16')
        train_data = ImageNet16(root, True , train_transform, 200)
        test_data  = ImageNet16(root, False, test_transform , 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, class_num
