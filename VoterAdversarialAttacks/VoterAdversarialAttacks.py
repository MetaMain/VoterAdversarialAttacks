import DefaultMethods
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    #Default Method Calls
    baseDir = "C://Users//kaleel//Desktop//Saved Models//Voter Dataset//Grayscale//"
    dirData = "C://Users//kaleel//Desktop//Kaleel//2024 Adversarial Voting//Voting Dataset//valLoaderGreyscaleBubbles.th"
    #DefaultMethods.CheckCleanAccAllModels(baseDir, dirData)
    attackNames = ["APGD-Original", "APGD", "PGD", "MIM", "FGSM"]
    modelNames = ["ResNet-20-B","ResNet-20-C","SimpleCNN-B", "SimpleCNN-C", "SVM-B", "SVM-C"]
    #Run the original APGD attack on ResNet-20-B
    DefaultMethods.MultiEpsAttack(baseDir, dirData, attackNames[0], modelNames[0])
    

if __name__ == '__main__':
    main()



