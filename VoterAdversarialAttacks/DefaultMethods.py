import torch
import numpy
import os
import math
from datetime import datetime
import DataManagerPytorch as DMP
import LoadVoterModel
import APGDVoter
import AttackWrappersWhiteBoxVoter
import APGDOriginal

#Run through the clean accuracy of all the models given the saved model folder annd
#the full path of the dataloader.
#baseDir = "C://Users//kaleel//Desktop//Saved Models//Voter Dataset//Grayscale//"
#dirData = "C://Users//kaleel//Desktop//Kaleel//2024 Adversarial Voting//DebugResNet//DebugResNet//valLoaderGreyscaleBubbles.th"
def CheckCleanAccAllModels(baseDir, dirData):   
    modelNames = ["ResNet-20-B","ResNet-20-C","SimpleCNN-B", "SimpleCNN-C", "SVM-B", "SVM-C"]
    #Setup the dataloader and device 
    valLoader = torch.load(dirData)
    device = torch.device('cuda')
    #Go through each model and load it in
    for i in range(0, len(modelNames)):
        model = LoadVoterModel.LoadModelGrayscaleByName(baseDir, modelNames[i])
        model.to(device)
        valAcc = DMP.validateD(valLoader, model, device)
        print("Val Acc " + modelNames[i] + ":", valAcc)

#The choice of models include: "ResNet-20-B","ResNet-20-C","SimpleCNN-B", "SimpleCNN-C", "SVM-B", "SVM-C".
#The choice of attacks include:
def MultiEpsAttack(baseDir, dirData, attackName, modelName):
    #Setup the model
    device = torch.device('cuda')
    model = LoadVoterModel.LoadModelGrayscaleByName(baseDir, modelName)
    model.to(device)
    model.eval()
    #Load the grayscale validation data
    valLoader = torch.load(dirData)
    numClasses = 2
    valAcc = DMP.validateD(valLoader, model, device)
    print("Val Acc " + modelName + ":", valAcc)
    #Get the clean loader 
    totalNumberOfSamplesRequired = 1000
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(model, totalNumberOfSamplesRequired, valLoader, numClasses)
    cleanAcc = DMP.validateD(cleanLoader, model, device)
    print("Clean Loader Acc "+ modelName + ":", cleanAcc)
    #Change batch size to 1 for the attack
    xClean, yClean = DMP.DataLoaderToTensor(cleanLoader)
    cleanLoader = DMP.TensorToDataLoader(xClean, yClean, transforms = None, batchSize =1, randomizer = None)
    clipMin = 0.0
    clipMax = 1.0
    #Setup folder for saving results 
    now = datetime.now()
    dtString = now.strftime("%m-%d-%Y,Hour(%H),Min(%M)")
    newpath=modelName+","+attackName+","+dtString
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    torch.save(cleanLoader, "CleanLoader,"+modelName+","+attackName)
    #Set the eps values represented in 0-1 
    epsMaxList = [4/255, 8/255, 16/255, 32/255, 64/255, 1.0]
    #Keep only 5 decimal places in the 
    for i in range(0, len(epsMaxList)):
        epsMaxList[i] = math.floor(epsMaxList[i]* 100000)/100000.0
    #Run through the attack with different eps values 
    for i in range(0, len(epsMaxList)):
        etaStart = 2*epsMaxList[i]
        #Run one of the white-box attacks based on name
        #APGD without gradient corrections
        if attackName == "APGD-Original": 
            numSteps = 100
            advLoader = APGDOriginal.AutoAttackPytorchMatGPUWrapper(device, cleanLoader, model, epsMaxList[i], etaStart, numSteps, clipMin, clipMax)
        #This is APGD with gradient corrections
        elif attackName == "APGD": 
             numSteps = 100
             advLoader = APGDVoter.AutoAttackPytorchMatGPUWrapper(device, cleanLoader, model, epsMaxList[i], etaStart, numSteps, clipMin, clipMax)
        #This is PGD with gradient corrections
        elif attackName == "PGD":
            numSteps = 60 
            epsilonStep = 0.008
            advLoader = AttackWrappersWhiteBoxVoter.PGDNativeAttackVoter(device, cleanLoader, model, epsMaxList[i], numSteps, epsilonStep, clipMin, clipMax)
        #This is MIM with gradient corrections
        elif attackName == "MIM":
            numSteps = 60 
            epsilonStep = 0.008
            decayFactor = 1.0
            advLoader = AttackWrappersWhiteBoxVoter.MIMNativeAttackVoter(device, cleanLoader, model, decayFactor, epsMaxList[i], numSteps, epsilonStep, clipMin, clipMax)
        #We run FGSM as a direct implementation of 1 step PGD
        elif attackName == "FGSM":
            numSteps = 1 
            epsilonStep = epsMaxList[i]
            advLoader = AttackWrappersWhiteBoxVoter.PGDNativeAttackVoter(device, cleanLoader, model, epsMaxList[i], numSteps, epsilonStep, clipMin, clipMax)
        else:
            raise ValueError("Attack name not recognized.")
        #Check the advLoader accuracy and save the adversarial samples
        advAcc = DMP.validateD(advLoader, model, device)
        print("Adv Acc "+modelName+"=", advAcc, "for " + attackName + " epsMax=", epsMaxList[i])
        #Save the adversarial samples for use later as a dataloader
        torch.save(advLoader, modelName+","+"epsMax="+str(epsMaxList[i])+","+attackName+".pt")
        xAdv, yAdv = DMP.DataLoaderToTensor(advLoader)
        #Also save the samples as a numpy array
        xAdvNumpy = xAdv.numpy()
        numpy.save(modelName+","+"epsMax="+str(epsMaxList[i])+","+attackName+".npy", xAdvNumpy)
