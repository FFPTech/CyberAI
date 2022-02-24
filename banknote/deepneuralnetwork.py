# Import relevant packages
import pandas as pd # Necessary for opening csv files
import numpy as np
import itertools # Necessary for plotting confusion matrix
from sklearn.metrics import confusion_matrix # Necessary for computing confusion matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # Import Multi-Layer Perceptron
from utils import plot_confusion_matrix
from sklearn import manifold


# STEP 1: Load dataset
banknote_datadset = pd.read_csv('data.csv')
X = banknote_datadset.iloc[:, 0:4].values # 4-dimensional input containing wavelet variance, skewness, curtosis and image entropy
y = banknote_datadset.iloc[:, 4].values # 1-dimensional output containing 'real' or 'fake' label
# tsne = manifold.TSNE(n_components=2, init="pca", random_state=0, verbose=1)
# Y = tsne.fit_transform(X)
# plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)
# plt.show()

# STEP 2: Define classification model
arch_big = (5, 50, 50, 50, 50)
arch_small = (5, 5)
mlp = MLPClassifier(hidden_layer_sizes=arch_small, 
    max_iter=10, alpha=1e-4, solver='adam', 
    verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.01)

# STEP 3: Train classficiation model
mlp.fit(X, y)
print("Training set score: %f" % mlp.score(X, y))
print("Test set score: %f" % mlp.score(X, y))

# STEP 4: TODO: Display confusion matrix (see section 21.6)
print('\n\nPrint confusion matrix')
class_names = ['fake', 'real']
y_pred = mlp.predict(X)
cm = confusion_matrix(y, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()