import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import math

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*3*3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.feature_dim = 2
        self.fc2 = nn.Linear(self.feature_dim, num_classes)
        self.name = 'LeNet++'


    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.resnet18(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        #self.base_model.fc = nn.Identity()
        #self.fc2 = nn.Linear(self.feature_dim,self.num_classes)
        #self.code_dim = 64
        #self.linear = nn.Linear(self.feature_dim, self.code_dim )
        #self.base_model.fc = nn.Sequential( nn.Dropout(), nn.Linear(self.feature_dim,self.num_classes) )   
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'ResNet18' 

    def forward(self,x):
        x = self.base_model(x)
        #y = self.fc2(x) 
        #y,x = self.base_model(x)
        #y = self.linear(y)
        #print(y.shape,x.shape)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.resnet50(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        #self.base_model.fc = nn.Identity()
        #self.fc2 = nn.Linear(self.feature_dim,self.num_classes)
        #self.code_dim = 64
        #self.linear = nn.Linear(self.feature_dim, self.code_dim )
        #self.base_model.fc = nn.Sequential( nn.Dropout(), nn.Linear(self.feature_dim,self.num_classes) )   
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'ResNet50' 

    def forward(self,x):
        x = self.base_model(x)
        #y = self.fc2(x) 
        #y,x = self.base_model(x)
        #y = self.linear(y)
        #print(y.shape,x.shape)
        return x


class MNISTResNet(nn.Module):
    def __init__(self,num_classes=10,in_channels=1):
        super(MNISTResNet,self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.feature_dim, num_classes)
        self.name = 'MNISTResNet18'
    
    def forward(self, x):
        return self.base_model(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.feature_dim = self.base_model.last_channel
        self.base_model.classifier = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'MobileNetV2' 

    def forward(self,x):
        x = self.base_model(x)
        return x



class Inception(nn.Module):
    def __init__(self, num_classes):
        super(Inception, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.inception_v3(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'InceptionV3' 

    def forward(self,x):
        x = self.base_model(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.vgg19(pretrained=True)
        #self.feature_dim = self.base_model.fc.in_features
        #self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
        self.base_model.classifier[-1] = nn.Linear(4096,self.num_classes)        
        self.name = 'VGG19' 

    def forward(self,x):
        x = self.base_model(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.resnet34(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        #self.base_model.fc = nn.Identity()
        #self.fc2 = nn.Linear(self.feature_dim,self.num_classes)
        #self.code_dim = 64
        #self.linear = nn.Linear(self.feature_dim, self.code_dim )
        #self.base_model.fc = nn.Sequential( nn.Dropout(), nn.Linear(self.feature_dim,self.num_classes) )   
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'ResNet34' 

    def forward(self,x):
        x = self.base_model(x)
        #y = self.fc2(x) 
        #y,x = self.base_model(x)
        #y = self.linear(y)
        #print(y.shape,x.shape)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super(ShuflleNet,self).__init__()
        #torch.hub.set_dir('/work/uah001')
        self.num_classes = num_classes
        self.base_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        #self.base_model.fc = nn.Identity()
        #self.fc2 = nn.Linear(self.feature_dim,self.num_classes)
        #self.code_dim = 64
        #self.linear = nn.Linear(self.feature_dim, self.code_dim )
        #self.base_model.fc = nn.Sequential( nn.Dropout(), nn.Linear(self.feature_dim,self.num_classes) )   
        self.base_model.fc = nn.Linear(self.feature_dim,self.num_classes)    
                
        self.name = 'ShuffleNet' 

    def forward(self,x):
        x = self.base_model(x)
        #y = self.fc2(x) 
        #y,x = self.base_model(x)
        #y = self.linear(y)
        #print(y.shape,x.shape)
        return x







__factory = {
    'cnn': ConvNet,
    'inception': Inception,
    'vgg': VGG,
    'net': Net,
    'resnet34':ResNet34,
    'resnet50':ResNet50,
    'mnistresnet': MNISTResNet,
    'mobilenet': MobileNet,
    'shufflenet': ShuffleNet,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    pass
