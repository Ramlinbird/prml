# Authors: Kyle Kastner
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
print ("Mean absolute unsigned error : %.6f" % err)
for X_transformed, title, k in [(X_ipca, "Incremental PCA", 1), (X_pca, "PCA", 2)]:
    plt.subplot(1, 2, k)
    for c, i, target_name in zip("rgb", [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    c=c, label=target_name)

    if "Incremental" in title:
        plt.title(title + " of iris dataset")
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best")
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()