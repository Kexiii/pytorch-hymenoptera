from __future__ import print_function, division

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
from datetime import datetime
import os
import copy
import json
import argparse
import pickle
from data_loader.data_loader import get_data_loaders
from utils.util import adjust_learning_rate
from utils.util import model_snapshot
from utils.util import ensure_dir
from model.resnet import get_resnet

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument('--resume',help='Resume from previous work')

def main():
    args = parser.parse_args()
    
    config = json.load(open("config.json"))
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    model = get_resnet().cuda()
    if args.resume is not None:
        print("Resume from previous work")
        model.load_state_dict(torch.load(args.resume))
    optimizer = optim.SGD(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    train(config,model,optimizer)
    
def train(config,model,optimizer):
    """Train the model
    Params:
        config:
            json config data
        model:
            model to train
        optimizer:
            optimizer used in training
    Return:
        None
    """
    data_loaders = get_data_loaders()
    ensure_dir(config['log_dir'])
    t_begin = time.time()
    best_acc, old_file = 0, None
    history = {'train':{'loss':[],'acc':[]},'val':{'loss':[],'acc':[]}}
    for epoch in range(config['epoch_num']):
        model.train() # train phase
        epoch_loss = 0
        epoch_correct = 0
        adjust_learning_rate(config,optimizer,epoch)
        for batch_idx, (data,target) in enumerate(data_loaders['train']):
            indx_target = target.clone()
            data, target = Variable(data.cuda()),Variable(target.cuda())
            optimizer.zero_grad()
            output = model(data)
            #define your own loss function here
            loss = F.cross_entropy(output,target)
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.cpu().eq(indx_target).sum()
            epoch_correct += correct
            
            if config['batch_log'] and batch_idx % config['batch_log_interval'] == 0 and batch_idx > 0:
                acc = correct * 1.0 / len(data)
                print('Train Epoch: {} [{}/{}] Batch_Loss: {:.6f} Batch_Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(data_loaders['train'].dataset),
                    loss.data[0], acc, optimizer.param_groups[0]['lr']))
        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(data_loaders['train'])
        eta = speed_epoch * config['epoch_num'] - elapse_time
        print("{}/{} Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(epoch+1,
            config['epoch_num'],elapse_time, speed_epoch, speed_batch, eta))
        
        
        epoch_loss = epoch_loss / len(data_loaders['train']) # average over number of mini-batch
        acc = 100. * epoch_correct / len(data_loaders['train'].dataset)
        print('\tTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch_loss, epoch_correct, len(data_loaders['train'].dataset), acc))
        history['train']['loss'].append(epoch_loss)  
        history['train']['acc'].append(acc)          
        model_snapshot(model, os.path.join(config['log_dir'], 'latest.pth'))
        
        if epoch % config['val_interval'] == 0:
            model.eval()
            val_loss = 0
            correct = 0
            for data, target in data_loaders['val']:
                indx_target = target.clone()
                data, target = Variable(data.cuda(),volatile=True),  Variable(target.cuda())
                output = model(data)
                val_loss += F.cross_entropy(output, target).data[0]
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

            val_loss = val_loss / len(data_loaders['val']) # average over number of mini-batch
            acc = 100. * correct / len(data_loaders['val'].dataset)
            print('\tVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                val_loss, correct, len(data_loaders['val'].dataset), acc))
            history['val']['loss'].append(val_loss)  
            history['val']['acc'].append(acc)
            if acc > best_acc:
                new_file = os.path.join(config['log_dir'], datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'-best-{}.pth'.format(epoch))
                model_snapshot(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file
    f = open(config['history'],'wb')
    try:
        pickle.dump(history,f)
    finally:
        f.close()
    print("Total Elapse: {:.2f}s, Best Val Acc: {:.3f}%".format(time.time()-t_begin, best_acc))
        
            
if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()