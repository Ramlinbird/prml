# Similar as LDA, need not assume same covariance between classes.
from sklearn.qda import QDA
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# Visualization
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(X[y==1, 0], X[y==1, 1], color='g') 
plt.scatter(X[y==2, 0], X[y==2, 1], color='b') 
plt.title('X Data Set Visualization')

# Classification
clf = QDA()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
plt.show()