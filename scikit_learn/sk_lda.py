from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print 'iris.data[-10:]\n', iris.data[-10:]
print 'iris.target[-10:]\n', iris.target[-10:]
print 'iris.data.shape:', iris.data.shape

from sklearn.lda import LDA

# Data dimension reduction
lda = LDA() # Default n_components setting, max C-1
lda_result1 = lda.fit_transform(iris.data, iris.target)
print 'LDA result 1:', lda_result1.shape
lda = LDA(n_components=1)
lda_result2 = lda.fit_transform(iris.data, iris.target)
print 'LDA result 2:', lda_result2.shape

# Visualization
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.scatter(lda_result1[iris.target==0, 0], lda_result1[iris.target==0, 1], color='r')
plt.scatter(lda_result1[iris.target==1, 0], lda_result1[iris.target==1, 1], color='g') 
plt.scatter(lda_result1[iris.target==2, 0], lda_result1[iris.target==2, 1], color='b') 
plt.title('LDA on iris (1)')

plt.subplot(1,2,2)
plt.stem(lda_result2)
plt.title('LDA on iris (2)')

plt.show()


# Classification
x_train_set = iris.data[:-5]
y_train_set = iris.target[:-5]
x_test_set = iris.data[-5:]
y_test_set = iris.target[-5:]
clf = LDA()
clf.fit(x_train_set, y_train_set)
y_pre = clf.predict(x_test_set)
print 'y_pre = \n', y_pre
print 'y_corret = \n', y_test_set
