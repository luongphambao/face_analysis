import torch
import torchvision
from torchvision.models import efficientnet_b0, efficientnet_b1,resnet18, resnet34, resnet50, resnet101
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class HydraNetModified(nn.Module):
        def __init__(self, net,backbone="resnet"):
            super(HydraNetModified, self).__init__()
            self.net = net
            self.backbone=backbone
            if self.backbone=="resnet":
                self.n_features = self.net.fc.in_features
                self.net.fc = nn.Identity()
                self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 6))]))
                self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 2))]))
                self.net.fc3 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 3))]))
                self.net.fc4 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 2))]))
                self.net.fc5 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 4))]))
                self.net.fc6 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 7))]))
            elif self.backbone=="efficientnet":
                classifier=self.net.classifier
                self.n_features = classifier[1].in_features
                self.net.classifier=nn.Identity()
                self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 6))]))
                self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 2))]))
                self.net.fc3 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 3))]))
                self.net.fc4 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 2))]))
                self.net.fc5 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 4))]))
                self.net.fc6 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('dropout', nn.Dropout(0.2)),('final', nn.Linear(self.n_features, 7))]))
        def forward(self, x):
            age_head = self.net.fc1(self.net(x))
            gender_head = self.net.fc2(self.net(x))
            race_head = self.net.fc3(self.net(x))
            masked_head = self.net.fc4(self.net(x))
            skintone_head = self.net.fc5(self.net(x))
            emotion_head = self.net.fc6(self.net(x))
            return age_head,gender_head,race_head,masked_head,skintone_head,emotion_head