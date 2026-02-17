"""统一数据加载模块"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CIFAR10Pair(Dataset):
    """生成对比学习所需的图像对"""
    def __init__(self, root='./data', train=True, transform=None):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=None
        )
        self.transform = transform or self.default_transform(train)
    
    @staticmethod
    def default_transform(train=True):
        """默认数据增强"""
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            return self.transform(img), self.transform(img), label
        return img, img, label

def get_dataloader(config, train=True):
    """获取数据加载器"""
    dataset = CIFAR10Pair(
        root=config['data']['root'],
        train=train,
        transform=None if not train else CIFAR10Pair.default_transform(train)
    )
    
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=train,
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True,
        drop_last=train
    )