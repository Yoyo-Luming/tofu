import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Resnet50(nn.Module):
    def __init__(self,hidden_dim=300):
        super(Resnet50, self).__init__()

        resnet = resnet50()
        self.base_model = resnet

        resnet.conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_model.conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=1, bias=False)
        modules=list(resnet.children())[:-1]
        self.main = nn.Sequential(*modules)
        # take the last layer representation from the resnet
        self.out_dim = resnet.fc.in_features
        self.fc = nn.Linear(2048, hidden_dim)
        

    def forward(self, x):
        # x = self.main(x)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return x.squeeze(-1).squeeze(-1)
        return x

    def domain_features(self, x):
        '''
        get domain features for dg_mmld
        '''
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        return x.view(x.size(0), -1)

    def conv_features(self, x) :
        '''
        get domain features for dg_mmld
        '''
        results = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # results.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        results.append(x)
        x = self.base_model.layer2(x)
        results.append(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # results.append(x)
        return results
