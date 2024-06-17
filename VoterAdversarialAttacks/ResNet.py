#This is Pytorch version of the ResNet V2 architecture copied over from Keras 
#Reference:
#[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#    Deep Residual Learning for Image Recognition. arXiv:1512.03385
#[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#If you use this implementation in you work, please don't forget to mention the
#author, Yerlan Idelbayev.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy

#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, res_block, activation, batch_normalization, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
        self.res_block = res_block
        self.activation = activation 
        self.batch_normalization = batch_normalization

        #Keras ResNetV2 architecture
        if res_block == 0:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv1 = nn.Conv2d(planes, in_planes, kernel_size=1, stride=stride, bias=True)

        self.bn2 = nn.BatchNorm2d(in_planes)
        #self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn3 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=True)

        #Skip connection setup here I think 
        self.shortcut = nn.Sequential()
        if res_block == 0:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        #The forward is slightly tricky to copy Keras implementation 
        if self.activation==True and self.batch_normalization ==True: #Fire everything 
            out =  self.conv1(F.relu(self.bn1(x))) #BN->RELU->CONV
        elif self.activation==True and self.batch_normalization ==False: #Activation, no batch normalization 
            out =  self.conv1(F.relu(x))
        elif self.activation==False and self.batch_normalization ==True: #No activation, batch normalization 
            out =  self.conv1(self.bn1(x))
        elif self.activation==False and self.batch_normalization ==False: #No activation, no batch normalization 
            out =  self.conv1(x)
        out =  self.conv2(F.relu(self.bn2(out)))
        out =  self.conv3(F.relu(self.bn3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, inputShape, dropOutRate = 0, numClasses=10):
        super(ResNet, self).__init__()
        self.numClasses = numClasses
        self.dropOutRate = dropOutRate
        self.in_planes = 16
        self.conv1 = nn.Conv2d(inputShape[1], 16, kernel_size=3, stride=1, padding=1, bias=True)
        #print("WARNING: NETWORK ONLY CONFIGURED FOR GRAYSCALE.")
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(16)
        #In PyTorch these are called layers, in Keras they are called stages the ResNets all have 3 stacks 
        #Each stack contains a certain number of blocks (6 in the ResNet56 implementation)
        #The first stack 
        stageNum = 0
        in_planes = 16
        self.layer1 = self._make_layer(stageNum, block, in_planes, num_blocks[0])
        #The second stack
        stageNum = 1
        in_planes = 64
        self.layer2 = self._make_layer(stageNum, block, in_planes, num_blocks[1])
        #The third stack
        stageNum = 2
        in_planes = 128
        self.layer3 = self._make_layer(stageNum, block, in_planes, num_blocks[2])

        #Classifer 
        classifierInputSize = in_planes * 2
        self.bn2 = nn.BatchNorm2d(classifierInputSize) #x = BatchNormalization()(x)
        
        #breaker = 5
        #if inputImageSize == 32 and num_blocks[0]==6:
        #    forwardInputSize = 256
        #elif inputImageSize ==224 and num_blocks[0]==6:
        #    forwardInputSize =12544
        #elif inputImageSize == 32 and num_blocks[0]==18:
        #    forwardInputSize =256 
        #else:
        #    raise ValueError("Input size not configured for the architecture. Compute the forward input size and recode around line 105.")
        #zz = torch.zeros(inputShape)
        #print(zz.shape)

        forwardInputSize = self.forwardDebug(torch.zeros(inputShape))
        self.sm = nn.Linear(in_features=forwardInputSize, out_features=numClasses)
        self.apply(_weights_init)
        
        # Sigmoid output replaces softmax
        #self.sigmoid = nn.Sigmoid()

        # Dropout layer
        self.drop = nn.Dropout(p = self.dropOutRate)

    def _make_layer(self, stageNum, block, in_planes, num_blocks):
        layers = []
        for res_block in range(0, num_blocks):  
            #This setup is almost all directly copied from Keras 
            activation = True
            batch_normalization = True
            strides = 1
            if stageNum == 0:
                planes = in_planes * 4
                if res_block == 0:  # first layer and first stage
                    activation = False
                    batch_normalization = False
            else:
                planes = in_planes * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
            #End of Keras parameter setup 
            layers.append(block(res_block, activation, batch_normalization, in_planes, planes, strides))
        return nn.Sequential(*layers)

    #Should ONLY be used for debugging purposes
    def forwardDebug(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, 8) #pool size 8
        out = out.view(out.size(0), -1) #This should replicate behavior of flatten
        return out.shape[1]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, 8) #pool size 8
        out = out.view(out.size(0), -1) #This should replicate behavior of flatten
        if self.dropOutRate != 0: out = self.drop(out)
        #out = self.sm(out)
        if self.numClasses == 1: 
            out = torch.sigmoid(out)
            #out = self.sigmoid(out)
        #out = F.softmax(self.sm(out))
        out = self.sm(out)
        return out

def resnet20(inputShape, dropOutRate, numClasses):
    return ResNet(BasicBlock, [2, 2, 2], inputShape, dropOutRate, numClasses)

def resnet56(inputShape, dropOutRate, numClasses):
    return ResNet(BasicBlock, [6, 6, 6], inputShape, dropOutRate, numClasses) #This is V2
    #return ResNet(BasicBlock, [9, 9, 9]) #This was V1 in Keras kind of

def resnet164(inputImageSize, dropOutRate, numClasses):
    return ResNet(BasicBlock, [18, 18, 18], inputImageSize, dropOutRate, numClasses)

def resnet1001(inputImageSize, dropOutRate, numClasses):
    return ResNet(BasicBlock, [111, 111, 111], inputImageSize, dropOutRate, numClasses)

