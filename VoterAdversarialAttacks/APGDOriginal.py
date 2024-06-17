#This is the original APGD attack that does not have any gradient masking fixes 
import torch
import DataManagerPytorch as DMP
import numpy as np

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#Function for computing the model gradient
def GetModelGradient(device, model, xK, yK):
    #Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    xK.requires_grad = True
    #Pass the inputs through the model 
    outputs = model(xK.to(device))
    model.zero_grad()
    #Compute the loss 
    cost = loss(outputs, yK)
    cost.backward()
    xKGrad = xK.grad
    return xKGrad

def ComputePList(pList, startIndex, decrement):
    #p(j+1) = p(j) + max( p(j) - p(j-1) -0.03, 0.06))
    nextP = pList[startIndex] + max(pList[startIndex] - pList[startIndex-1] - decrement, 0.06)
    #Check for base case 
    if nextP>= 1.0:
        return pList
    else:
        #Need to further recur
        pList.append(nextP)
        ComputePList(pList, startIndex+1, decrement)

#Condition two checks if the objective function and step size previously changed
def CheckConditionTwo(f, eta, checkPointIndex, checkPoints):
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    if eta[previousCheckPoint] == eta[currentCheckPoint] and f[previousCheckPoint] == f[currentCheckPoint]:
        return True
    else:
        return False

#Condition one checks the summation of objective function
def CheckConditionOne(f, checkPointIndex, checkPoints, targeted):
    sum = 0
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    #See how many times the objective function was growing bigger 
    for i in range(previousCheckPoint, currentCheckPoint): #Goes from w_(j-1) to w_(j) - 1
        if f[i+1] > f[i] :
            sum = sum + 1
    ratio = 0.75 * (currentCheckPoint - previousCheckPoint)
    #For untargeted attack we want the objective function to increase
    if targeted == False and sum < ratio: #This is condition 1 from the Autoattack paper
        return True
    elif targeted == True and sum > ratio: #This is my interpretation of how the targeted attack would work (not 100% sure)
        return True
    else:
        return False

def ComputeCheckPoints_New(Niter, decrement, opt=False):
    #First compute the pList based on the decrement amount
    pList = [0, 0.22] #Starting pList based on AutoAttack paper
    ComputePList(pList, 1, decrement)
    #Second compute the checkpoints from the pList
    wList = []
    for i in range(0, len(pList)):
        wList.append(int(np.ceil(pList[i]*Niter)))
    #There may duplicates in the list due to rounding so finally we remove duplicates
    wListFinal = []
    for i in wList:
        if i not in wListFinal:
            wListFinal.append(i)
    #Return the final list
    return wListFinal, {k: v for v, k in enumerate(wListFinal)} if opt else wListFinal

def AutoAttackPytorchMatGPUWrapper(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1):
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    model.eval() #Change model to evaluation mode for the attack 
    #Go through each batch and run the attack
    for xData, yData in dataLoader:
        #Initialize the AutoAttack variables
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize #Update the tracking variable 
        print(tracker, end = "\r")
        xBest = AutoAttackPytorchMatGPU(device, xData, yData.long(), model, epsilonMax, etaStart, numSteps, clipMin, clipMax)
        xAdv[tracker-batchSize: tracker] = xBest
        yClean[tracker-batchSize: tracker] = yData
    advLoader = DMP.TensorToDataLoader(xAdv, yClean.long(), transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def AutoAttackPytorchMatGPU(device, xData, yData, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1): ### only for 1 batch and opt for memory
    #Setup attack variables:
    decrement = 0.03
    wList, wListIndex = ComputeCheckPoints_New(numSteps, decrement, True) #Get the list of checkpoints based on the number of iterations 
    alpha = 0.75 #Weighting factor for momentum 

    # model.eval() #Change model to evaluation mode for the attack 
    batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
    xShape = xData[0].shape
    
    xData = xData.to(device)
    yK = yData.to(device) #Correct class labels which don't change in the iterations
    eta = torch.zeros(numSteps + 1, batchSize).to(device) #Keep track of the step size for each sample
    eta[0, :] = etaStart #Initalize eta values as the starting eta for each sample in the batch 
    f = torch.zeros(numSteps + 1 , batchSize).to(device) #Keep track of the function value for every sample at every step
    x = torch.zeros(3, batchSize, xShape[0], xShape[1], xShape[2]).to(device)
    x[0] = xData #Initalize the starting adversarial example as the clean example
    lossIndividual = torch.nn.CrossEntropyLoss()
    
    #Do the attack for a number of steps
    for k in range(0, numSteps):
        #First attack step handled slightly differently
        if k == 0:
            xKGrad = GetModelGradient(device, model, x[0], yK) #Get the model gradient 
            x[1] = x[0] + eta[0][:, None, None, None] * torch.sign(xKGrad) #here we use index 1 because the 0th index is the clean sample
            x[1] = torch.clamp(ProjectionOperation(x[1], x[0], epsilonMax), min=clipMin, max=clipMax) #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                
            #Check which adversarial x is better, the clean x or the new adversarial x 
            with torch.no_grad():
                outputsOriginal = model(x[0].to(device)) 
                f[0] = lossIndividual(outputsOriginal, yK).detach() #Store the value in the objective function array
                outputs = model(x[1].to(device)) 
                f[1] = lossIndividual(outputs, yK).detach() #Store the value in the objective function array
                    
            values, indices = torch.max(f[0:2], dim=0)
            xBest = torch.stack([x[indices[i],i] for i in range(batchSize)])
            fBest = values
            #Give a non-zero step size for the next iteration
            eta[1] = eta[0]
                
        #Not the first iteration of the attack
        else:
            xKGrad = GetModelGradient(device, model, x[1], yK) 
            #Compute zk
            z = x[1] + eta[k][:, None, None, None] * torch.sign(xKGrad)
            z = ProjectionOperation(z, xData, epsilonMax)
            #Compute x(k+1) using momentum
            x[2] = x[1] + alpha *(z-x[1]) + (1-alpha)*(x[1]-x[0])
            x[2] =  ProjectionOperation(x[2], xData, epsilonMax)          
            #Apply the clipping operation to make sure xAdv remains in the valid image range
            x[2] = torch.clamp(x[2], min=clipMin, max=clipMax)
        
            #Check which x is better
            with torch.no_grad():
                outputs = model(x[2].to(device))
                f[k + 1] = lossIndividual(outputs, yK).detach()
                
            for b in range(0, batchSize):
                #In the untargeted case we want the cost to increase
                if f[k+1, b] >= fBest[b]: 
                    xBest[b] = x[2, b]
            fBest = torch.maximum(f[k + 1],fBest)
            
            #Now time to do the conditional check to possibly update the step size 
            if k in wListIndex: 
                checkPointIndex = wListIndex[k] #Get the index of the currentCheckpoint
                #Go through each element in the batch 
                for b in range(0, batchSize):
                    conditionOneBoolean = CheckConditionOne(f[:,b], checkPointIndex, wList, False)
                    conditionTwoBoolean = CheckConditionTwo(f[:,b], eta[:,b], checkPointIndex, wList)
                    #If either condition is true halve the step size, else use the step size of the last iteration
                    if conditionOneBoolean == True or conditionTwoBoolean == True:           
                        eta[k + 1, b] = eta[k, b] / 2.0
                    else:
                        eta[k + 1, b] = eta[k, b]
            #If we don't need to check the conditions, just repeat the previous iteration's step size
            else:
                eta[k + 1] = eta[k] 
            
            #Save x[k] to x[k-1], x[k+1] to x[k] for the next k
            x[0],x[1] = x[1],x[2]
        #Memory clean up
        torch.cuda.empty_cache() 
    return xBest