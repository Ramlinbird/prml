Follow are the using examples, you can directly type them in your cmd if you have set the python enviroment correctly. If you have any question or suggestions, please contact with me. Thanks for your watching!

```
python
import treepredict

# Test src data [Important]
treepredict.my_data

# Test divideset
treepredict.divideset(treepredict.my_data, 3, 18)

# Test giniimpurity
treepredict.giniimpurity(treepredict.my_data)

# Test entropy
treepredict.entropy(treepredict.my_data)

# Test buildtree
tree = treepredict.buildtree(treepredict.my_data)

# Test printtree
treepredict.printtree(tree)

# Test drawtree
treepredict.drawtree(tree, jpeg='tree1.jpg')

# Test prune
treepredict.prune(tree, 0.1)
treepredict.printtree(tree)
treepredict.prune(tree, 1.0)
treepredict.printtree(tree)
treepredict.drawtree(tree, jpeg='tree2.jpg')

# Test classify
treepredict.classify(['google', None, 'no', 17], tree)

# Test mdclassify
treepredict.mdclassify(['google', None, None, 17], tree)
treepredict.mdclassify(['google', None, None, None], tree)



# Modeling Home Prices
# This is an example which shows the explanatory ability of decision tree
import zillow
housedata = zillow.getpricelist()
housetree = treepredict.buildtree(housedata, scoref=treepredict.variance)
treepredict.drawtree(housetree, 'housetree.jpg')



# Modeling "Hotness"
# HTTP Error, if you figure it out, please contact me. Thank you!
import hotornot
l1 = hotornot.getrandomratings(500)
pdata = hotornot.getpeopledata(l1)
hottree = treepredict.buildtree(pdata, scoref=treepredict.variance)
treepredict.prune(hottree, 0.5)
treepredict.drawtree(hottree, 'hottree.jpg')
```
