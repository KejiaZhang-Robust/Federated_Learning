import copy
import torch
from torchvision import datasets, transforms

from samples import cifar_iid, cifar_noniid, mnist_iid, mnist_noniid

def create_dataloader(config):
    if config.DATA.Dataset == 'CIFAR10':
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=apply_transform)

        # sample training data amongst users
        if config.DATA.Split == 'IID':
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, config.Train.Num_users)
        elif config.DATA.Split == 'NonIID':
            user_groups = cifar_noniid(train_dataset, config.Train.Num_users)
            
    elif config.DATA.Dataset == 'Mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if config.DATA.Split == 'IID':
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, config.Train.Num_users)
        elif config.DATA.Split == 'NonIID':
            # Chose euqal splits for every user
            user_groups = mnist_noniid(train_dataset, config.Train.Num_users)


    return train_dataset, test_dataset, user_groups
