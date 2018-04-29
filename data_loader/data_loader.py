from __future__ import print_function, division

__all__ = ["get_data_loader"]

import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import customized_transforms as ct
import os
import json
import copy

# Define your data loader config path here
data_loader_config_path = os.path.join(os.path.dirname(__file__),"config.json")

# Define your own data transforms here
data_transforms = {
    'train': transforms.Compose([
        ct.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        ct.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        ct.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_data_loaders():
    """Get the pytorch style data loader
    Return:
        Pytorch style data loader
    Note:
        Currently only train and validation phase is implemented, you can 
        implement the test phase in the same way
    """
    config = json.load(open(data_loader_config_path))
    
    data_dir = os.path.join(os.path.dirname(__file__),config['data_dir'])
    # Uncomment this if you use the absolute path
    # data_dir = config['data_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                        data_transforms[x]) for x in ('train','val')}
    data_loaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,
                     shuffle=True, num_workers=num_workers) for x in ('train', 'val')}
    return data_loaders


def imshow(input, title=None):
    """Visualize the pics after data augmentation
    Params:
        input:
            pytorch image batch tensor with shape (batch_size,color_channel,W,H) 
        title:
            title for the output image
    Return:
        None
    """
    input = torchvision.utils.make_grid(input)
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.savefig("pics_after_data_augmentation.jpg")
    
    
def test_data_loader(pics_num = 2):
    """Test our data loader and visualize some of the pics after data augmention
    Params:
        pics_num:
            How many pics you want to observe, should be less
            than the batch_size in config.json
    Return:
        None
    """
    config = json.load(open(data_loader_config_path))
    data_dir = config['data_dir']
    data_loaders = get_data_loaders()
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                        data_transforms[x]) for x in ('train','val')}
    class_names = image_datasets['train'].classes
    input, classes = next(iter(data_loaders['train']))
    imshow(input[0:pics_num], title=[class_names[x] for x in classes[0:pics_num]])
    
    
if __name__ == "__main__":
    test_data_loader()
