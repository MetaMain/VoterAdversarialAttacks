#The PGD and MIM attacks which also applies gradient correction if there is zero gradient and the class is recognized correctly
import torch
import DataManagerPytorch as DMP

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#This is PGD attack with special case for the grayscale voter dataset.
#Works like normal PGD UNLESS the gradient is zero and class label is correct. Then the image is either brightened or darkened instead of moving in 
#the gradient step direction. This allows us avoid the vanishing graident and keep perturbing the image.
def PGDNativeAttackVoter(device, dataLoader, model, epsilonMax, numSteps, epsilonStep, clipMin, clipMax):
    model.eval()  #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize
        print(tracker, end = "\r")
        #print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        for attackStep in range(0, numSteps):
            xGrad, zeroGradFlag = GetModelGradient(device, model, xAdvCurrent, yCurrent)
            advTemp = xAdvCurrent + (epsilonStep * xGrad.sign()).to(device)
            advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader
    return advLoader

#Function for computing the model gradient
#Also returns a zero grad flag 
def GetModelGradient(device, model, xK, yK):
    #Flag for the zero gradient condition
    zeroGradFlag = False
    #Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    xK.requires_grad = True
    #Pass the inputs through the model 
    outputs = model(xK.to(device))
    indexMax = torch.argmax(outputs).item()
    model.zero_grad()
    #Compute the loss 
    cost = loss(outputs, yK)
    cost.backward()
    xKGrad = xK.grad
    #Here is the main change to APGD, check to make sure the gradient is not 0.
    xGrad = xK.grad.data.to("cpu")
    maxGrad = torch.max(torch.abs(xGrad[0])).item()
    if maxGrad == 0 and (indexMax == yK[0].item()):
        #Set the zeroGradFlag so we know we reached this case
        zeroGradFlag = True
        if yK[0].item()==0: #Black so vote class
            #print("hit case 0")
            target = torch.ones(xK.shape).to(device)
            gx = (target-xK.detach())
            return gx, zeroGradFlag
        if yK[0].item()==1: #white so non-vote class
            #print("hit case 1")
            target = torch.ones(xK.shape).to(device)
            gx = -1.0*(target-xK.detach()) #Make whole image darker
            return gx, zeroGradFlag
    else:
        return xKGrad, zeroGradFlag

#When we hit the zero gradient case check if a smaller eps bounds will help us get a gradient
#This returns the adversarial example WITH projection applied so output will always be less than or equal to epsMax bounds
def GradientLookAheadwithProjection(xK, yK, xData, model, epsilonMax, device):
    #Arbitrarily create a range for epsMax bounds
    epsBoundList = []
    #We do reverse range because if all have zero gradient then the widest projection will end up being used
    for i in range(5, 0, -1):
        #Generate a bunch of smaller bounds to test
        epsBoundList.append(epsilonMax/i) 
    #Go through and find the eps that gives the best gradient for the next step
    #Note we cannot easily parallelize this because of the limitations of PyTorch (we need indiviual gradients not batching)
    loss = torch.nn.CrossEntropyLoss()
    bestSumGrad = -1 #The best grad absolute value grad should always be 0 OR better so initialize to -1
    bestXAdv = None #This will always get filled in on the first run
    bestBound = -1
    for i in range(0, len(epsBoundList)):
        #Apply the projection with an eps equal to or less than epsMax 
        xAdv = ProjectionOperation(xK, xData, epsBoundList[i]).to(device)
        #Do single sample backprop to preserve the individual gradient 
        xAdv.requires_grad = True
        outputs = model(xAdv)
        cost = loss(outputs, yK)
        cost.backward()
        xAdvGrad = xAdv.grad.data.to("cpu")
        sumGrad = torch.sum(torch.abs(xAdvGrad[0])).item()
        #If we find a better gradient use it
        if sumGrad >=  bestSumGrad:
            bestSumGrad = sumGrad
            bestXAdv = xAdv.detach().clone().to("cpu") #Remove from the GPU and copy
            bestBound = epsBoundList[i]
    #print("Best Bound was:", bestBound)
    #Return the best adversarial example
    #Here best means "gives the biggest gradient for the next step" as we want to escape the zero gradient trap
    return bestXAdv

#MIM native attack with the zero gradient correction included 
def MIMNativeAttackVoter(device, dataLoader, model, decayFactor, epsilonMax, numSteps, epsilonStep, clipMin, clipMax):
    model.eval()  #Change model to evaluation mode for the attack
    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize
        print(tracker, end = "\r")
        #print("Processing up to sample=", tracker)
        # Put the data from the batch onto the device
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        for attackStep in range(0, numSteps):
            xGrad, zeroGradFlag = GetModelGradient(device, model, xAdvCurrent, yCurrent)
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xGrad)
            advTemp = xAdvCurrent + (epsilonStep * torch.sign(gMomentum)).to(device)
            advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader
    return advLoader

def GradientNormalizedByL1(gradient):
    #Do some basic error checking first
    #if gradient.shape[1] != 3:
    #    raise ValueError("Shape of gradient is not consistent with an RGB image.")
    #basic variable setup
    batchSize = gradient.shape[0]
    colorChannelNum = gradient.shape[1]
    imgRows = gradient.shape[2]
    imgCols = gradient.shape[3]
    gradientNormalized = torch.zeros(batchSize, colorChannelNum, imgRows, imgCols)
    #Compute the L1 gradient for each color channel and normalize by the gradient 
    #Go through each color channel and compute the L1 norm
    for i in range(0, batchSize):
        for c in range(0, colorChannelNum):
           norm = torch.linalg.norm(gradient[i,c], ord=1)
           gradientNormalized[i,c] = gradient[i,c]/norm #divide the color channel by the norm
    return gradientNormalized

