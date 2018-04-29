from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os
import json
from data_loader.data_loader import get_data_loaders
from model.resnet import get_resnet

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument('--checkpoint',required = True,help='Checkpoint path')

def main():
    args = parser.parse_args()
    config = json.load(open("config.json"))
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    model = get_resnet().cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    test(config,model)
    
def test(config,model):
    """Train the model
    Params:
        config:
            json config data
        model:
            model to train
    Return:
        None
    """
    data_loaders = get_data_loaders()
    model.eval()
    val_loss = 0
    correct = 0
    #Note, we still use the validation set to test here, you should change it to test set in your project
    for data, target in data_loaders['val']:
        indx_target = target.clone()
        data, target = Variable(data.cuda(),volatile=True),  Variable(target.cuda())
        output = model(data)
        val_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

    val_loss = val_loss / len(data_loaders['val']) # average over number of mini-batch
    acc = 100. * correct / len(data_loaders['val'].dataset)
    print('Resutls: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        val_loss, correct, len(data_loaders['val'].dataset), acc))
          
if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()