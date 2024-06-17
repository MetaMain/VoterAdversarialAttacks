import torch
import DataManagerPytorch as DMP
import torch.nn as nn
import torch.nn.functional as F

class PseudoSVM(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.layer = torch.nn.Linear(insize, outsize, bias = True)
        self.sigmoid = False
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        if self.sigmoid:
            return self.s(self.layer(x)).T[0]
        return self.layer(x).T[0]

class PseudoTwoOutputSVM(torch.nn.Module):
    def __init__(self, insize, dir):
        super().__init__()
        self.startingModel = None
        self.layer = torch.nn.Linear(in_features = insize, out_features = 1, bias = True)
        self.layer2 = torch.nn.Linear(in_features = 1, out_features = 2, bias = True)
        #Setup the weights 
        self.LoadMultiOutputSVM(dir)

    def forward(self, x):
        x = 255 * torch.flatten(x, start_dim = 1) 
        out = self.layer(x)
        return self.layer2(out)

    #Hardcoded for grayscale by researcher K
    def LoadMultiOutputSVM(self, dir):
        inputDim = 2000     
        checkpoint = torch.load(dir)
        model = PseudoSVM(insize = inputDim, outsize = 1)
        model.load_state_dict(checkpoint)
        with torch.no_grad():
            self.layer.weight = nn.Parameter(model.layer.weight)
            self.layer.bias[0] = model.layer.bias.item()
            self.layer2.weight[0, 0] = -1.
            self.layer2.weight[1, 0] = 1.
            self.layer2.bias[0] = 1.
            self.layer2.bias[1] = 0.
