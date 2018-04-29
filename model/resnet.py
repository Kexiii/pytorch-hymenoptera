import torchvision.models as models
import torch.nn as nn

def get_resnet():
    model  = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,2)
    return model