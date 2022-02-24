# Import relevant packages
import pandas as pd # Necessary for opening csv files
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
import itertools # Necessary for plotting confusion matrix
from sklearn.metrics import confusion_matrix # Necessary for computing confusion matrix
import matplotlib.pyplot as plt

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
print('Exercise 3: SVM on UCI Datasets')
print('===============================')

# STEP 1: Load dataset
banknote_datadset = pd.read_csv('data_banknote_authentication.csv')
X = banknote_datadset.iloc[:, 0:4].values # 4-dimensional input containing wavelet variance, skewness, curtosis and image entropy
y = banknote_datadset.iloc[:, 4].values # 1-dimensional output containing 'real' or 'fake' label

# STEP 2: Define classification model
svc = svm.SVC(kernel='linear') # Try other kernels, i.e. 'linear', 'poly', 'rbf', 'sigmoid' or 'precomputed'

# STEP 3: Train classficiation model
C_s = np.logspace(-10, 0, 10) # Regularization parameter: Exponent space C = 10^(-10) until 10^(0) 
scores = list()
scores_std = list()
for C in C_s: # Loop over regularization parameter
    print('Train SVM with regularization parameter C = {0}'.format(C))
    svc.C = C
    this_scores = cross_val_score(svc, X, y, n_jobs=1) # train SVC and obtain multiple scores
    scores.append(np.mean(this_scores)) # Calculate mean score
    scores_std.append(np.std(this_scores)) # Calculate std score

# STEP 4: Display classification results
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.semilogx(C_s, scores) # Plot mean (average) scores in function of regularization parameter C
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--') # Plot -sigma deviation around average
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--') # Plot +sigma deviation around average
locs, labels = plt.yticks() # Set axes
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score') # Set label for x-axis
plt.xlabel('Parameter C') # Set label for y-axis
plt.ylim(0, 1.1) # Set y-axis range
# plt.show() # Plot resulting graph

# STEP 5: TODO: Chose the best Kernel + Regularization paremeters for training the SVM
best_kernel = 'poly' # Set the best kernel type ('linear', 'poly', 'rbf', 'sigmoid')
best_C = 100 # Set the best regularization ârameter
print('\n\nTrain SVM with kernel {0} and regularization parameter C = {1}'.format(best_kernel, best_C))
best_svc = svm.SVC(kernel=best_kernel, C=best_C) # Setup classifier hyper-parameters
best_svc.fit(X, y) # train SVC and obtain multiple scores
y_pred = best_svc.predict(X)

# STEP 6: TODO: Display confusion matrix (see section 21.6)
print('\n\nPrint confusion matrix')
class_names = ['fake', 'real']
cm = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()