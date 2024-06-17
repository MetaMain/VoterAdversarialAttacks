import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from collections import OrderedDict

#Class that defines one same convolutional layer
class _Same_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3, 3), stride = 1)
        self.relu1 = torch.nn.ReLU(inplace = True)
        self.max1 = torch.nn.MaxPool2d(kernel_size = (2, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.max1(out)
        return out

class SimpleCNN(torch.nn.Module):
    # This is a very simple model architecture with only one dense block (same conv) that allows us to control how many layers/parameters are in our model
    # Voter Lab SVM can achieve around 88% accuracy, we want to find our how large CNNs have to be to reach similar accuracy
    # Inspired by this model: https://keras.io/examples/vision/mnist_convnet/
    def __init__(self, imgSize, blocks, dropOutRate = 0.9, numClasses = 1):
        super(SimpleCNN, self).__init__()
        self.blocks = blocks
        self.numClasses = numClasses

        # Initialize first block with imgSize[0] input channels
        self.features = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.Conv2d(in_channels = imgSize[0], out_channels = blocks[0], kernel_size = (3, 3), stride = 1)),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("max0", nn.MaxPool2d(kernel_size = (2, 2)))
                ])
            )
        # Add rest of layers to features
        for i in range(0, len(self.blocks)-1):
            block = _Same_Conv_Layer(in_channels = blocks[i], out_channels = blocks[i+1])
            self.features.add_module("denseblock" + str(i+1), block)

        # Dropout, FC, then output one unit
        self.drop0 = torch.nn.Dropout(p = dropOutRate)
        testInput = torch.zeros((1, imgSize[0], imgSize[1], imgSize[2]))
        outputShape = self.figureOutFlattenedShape(testInput)
        self.forward0 = torch.nn.Linear(in_features=outputShape[1], out_features=numClasses)

    def forward(self, x):
        out = x 
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.drop0(out)
        if self.numClasses == 1: out = torch.sigmoid(self.forward0(out))
        out = self.forward0(out)
        return out
    
    def figureOutFlattenedShape(self, testInput):
        out = testInput
        out = self.features(out)
        out = out.view(out.size(0), -1)
        return out.shape
    
# Return 30,000 same conv Simple CNN
def BuildSimpleCNN(imgSize, dropOutRate, numClasses): 
    return SimpleCNN(imgSize = imgSize, blocks = [32, 48, 32], dropOutRate = dropOutRate, numClasses = numClasses)
