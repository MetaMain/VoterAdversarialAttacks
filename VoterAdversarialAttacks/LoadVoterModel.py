#Class for loading all of the voter models based on name 
import torch
import ResNet
import SimpleCNN
import VoterSVM

#baseDir ="C://Users//kaleel//Desktop//Saved Models//Voter Dataset//Grayscale//"
def LoadModelGrayscaleByName(baseDir, modelName):
    #Shared parameter setup
    numClasses = 2
    imgSize = (1, 1, 40, 50)
    #Load a ResNet
    if modelName == "ResNet-20-B" or modelName == "ResNet-20-C":
        dropOutRate = 0.5 
        model = ResNet.resnet20(imgSize, dropOutRate, numClasses)  
    #Load a SimpleCNN
    elif modelName == "SimpleCNN-B" or modelName == "SimpleCNN-C":
        imgSize = imgSize[1:] #This is the size for the CNN that doesn't include batching
        dropOutRate = 0.9 
        model = SimpleCNN.BuildSimpleCNN(imgSize, dropOutRate, numClasses)
    #Load an SVM
    elif modelName == "SVM-B" or modelName == "SVM-C":
        inputDim = 2000    
        model = VoterSVM.PseudoTwoOutputSVM(inputDim, baseDir+modelName+".torch")
        return model #For the SVM we directly return as model loading is handled by the class
    else:
        raise ValueError("We don't have that type of model configured.")
    #For non-SVM models we load the saved weights and return 
    dirModel = baseDir + modelName
    checkpoint = torch.load(dirModel+".pth")
    model.load_state_dict(checkpoint['state_dict'])
    return model
