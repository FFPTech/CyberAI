# Import relevant packages
import pandas as pd # Necessary for opening csv files
import numpy as np
import itertools # Necessary for plotting confusion matrix
from sklearn.metrics import confusion_matrix # Necessary for computing confusion matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import *

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ----------------------------------------------------------
print('Exercise: Naive Bayes on UCI Datasets')
print('=====================================')

# STEP 1: Load dataset
banknote_datadset = pd.read_csv('data_banknote_authentication.csv')
X = banknote_datadset.iloc[:, 0:4].values # 4-dimensional input containing wavelet variance, skewness, curtosis and image entropy
y = banknote_datadset.iloc[:, 4].values # 1-dimensional output containing 'real' or 'fake' label

# STEP 2: Define classification model
nb = GaussianNB()

# STEP 3: Train classficiation model
nb.fit(X, y)
print("Training set score: %f" % nb.score(X, y))

# STEP 4: Display confusion matrix
print('\n\nPrint confusion matrix')
class_names = ['fake', 'real']
y_pred = nb.predict(X)
cm = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()