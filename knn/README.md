This time, I talk about KNN algorithm. And all you should do is following the python commands. Hope that you enjoy it!

```
python
# construct kdtree, example of 2-dims data
import kdtree
point_list = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
tree = kdtree.kdtree(point_list)
print(tree)

# Produce wineprice dataset manually
import numpredict
data = numpredict.wineset1()
print data[0]

# Test euclidean
print data[0]['input']
print data[1]['input']
numpredict.euclidean(data[0]['input'], data[1]['input'])

# Estimate price using knn's average price, default k = 5
numpredict.knnestimate(data, (99.0, 3.0), k = 5)
# compare with real hidden model's value
numpredict.wineprice(99.0, 3.0)

# Different weight setting according to the distance
numpredict.subtractweight(0.1)
numpredict.inverseweight(0.1)
numpredict.gaussian(0.1)
numpredict.subtractweight(1)
numpredict.inverseweight(1)
numpredict.gaussian(5)

# Test WeightedKnn
numpredict.WeightedKnn(data,(99.0,3.0))

# Compare weightedknn and knnestimate using cross-validation, also different k setting
numpredict.crossvalidation(numpredict.knnestimate, data)
numpredict.crossvalidation(numpredict.WeightedKnn, data)
def knn3(d,v): return numpredict.WeightedKnn(d,v,k=3)
numpredict.crossvalidation(knn3, data)

# Reconstruct dataset which have some differ-scales variables
data = numpredict.wineset2()
numpredict.crossvalidation(knn3, data)

# Rescale attribute's values manually
sdata = numpredict.rescale(data, [1, 1, 0, 0.1])
numpredict.crossvalidation(knn3, sdata)

# Using optimization.py in chapter5 to find paras automatically
import optimization
costf = numpredict.createcostfunc(knn3, data)
optimization.annealingoptimize(numpredict.weightdomain, costf, step = 2)
# [7.0, 11, 2, 13], [7.0, 7.0, 0, 1.0], results look very different if we run times, something wrong may exist

# Ebaypredict example, Not tried
import ebaypredict
laptops=ebaypredict.doSearch('laptop')
laptops[0:10]

```