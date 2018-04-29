import os
import shutil
import torch

def adjust_learning_rate(config,optimizer, epoch):
    """Define your own learning rate adjuting strategy here
    """
    lr = config['lr'] * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)
        
def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))
        
def model_snapshot(model, new_file, old_file=None, verbose=False):
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))
    torch.save(model.state_dict(), expand_user(new_file))
