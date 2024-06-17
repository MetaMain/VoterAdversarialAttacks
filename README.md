# Voter Adversarial Attacks

This is the Github repository for adversarial attacks on machine learning models trained on ballot data. At this time we cannot publicly provide the trained models or the dataset.

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>In the file "VoterAdversarialAttacks.py", change line 7 to point to your base directory where all voter models are saved.</li>
  <li>In the file "VoterAdversarialAttacks.py", change line 8 to point to your data directory where your ballot dataset is saved as a ".th" file.</li>
  <li>In the file "VoterAdversarialAttacks.py", change line 13 to run whichever attack and model you want. Lines 10 and 11 of this file list all available attacks ("APGD-Original", "APGD", "PGD", "MIM", "FGSM") and models ("ResNet-20-B","ResNet-20-C","SimpleCNN-B", "SimpleCNN-C", "SVM-B", "SVM-C").</li>
  <li>Run "VoterAdversarialAttacks.py". The output will be printed to the terminal and the corresponding adversarial examples will be saved as both a PyTorch dataloader and ".npy" file.</li>
</ol>


# Software Installation 

We use the following software packages: 
<ul>
  <li>pytorch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>numpy==1.19.2</li>
</ul>

# System Requirements 

All our attacks are tested in Windows 10 with 12 GB GPU memory (Titan V GPU).
# Contact
For questions or concerns please contact the author at: kaleel.mahmood@uconn.edu
