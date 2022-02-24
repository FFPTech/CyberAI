#STEP 0: Import Packages
import arff
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix

# STEP 1: Data Management
#========================

#Data Collection
data = []
rows = arff.load('data.arff')
for row in rows:   
    val = [float(v) for v in row._values]
    data.append(val) 
    # print(val)
data = np.array(data)
X = data[:, 1:31]
y = data[:, 30]

# STEP 2: Model Training
#=======================

# STEP 3: Model Evaluation
#=========================

print('Done')