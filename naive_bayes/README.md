Naive bayes is often used to solve document filtering problem, like classfying whether an email is trash. Following, we will give the running tutor of such naive byes classfier. What's more, here we also give one other famous method named **Fisher Method** which is particularly for spam filtering.

> HINTS: docclass.py and docclass_sqlite.py are nearly the say, but the storage method is differ. [docclass_sqlite.py NOT TESTED!]

```python

# Define basic classifier which contains basic info
import docclass
cl=docclass.classifier(docclass.getwords)
cl.train('the quick brown fox jumps over the lazy dog','good')
cl.train('make quick money in the online casino','bad')
cl.fcount('quick','good')
cl.catcount('bad')
cl.totalcount()
cl.categories()

# TEST fprob function
cl=docclass.classifier(docclass.getwords)
docclass.sampletrain(cl)
cl.fprob('quick','good')

# TEST weightedprob function which calculating probs starting with a Reasonable Guess
cl=docclass.classifier(docclass.getwords)
docclass.sampletrain(cl)
cl.weightedprob('money','good',cl.fprob)

# Naive Bayes Prob cal
cl=docclass.naivebayes(docclass.getwords)
docclass.sampletrain(cl)
cl.prob('quick rabbit','good')
cl.prob('quick rabbit','bad')

# Naive Bayes classifier
cl=docclass.naivebayes(docclass.getwords)
docclass.sampletrain(cl)
cl.classify('quick rabbit',default='unknown')
cl.classify('quick money',default='unknown')
cl.setthreshold('bad',3.0)
cl.classify('quick money',default='unknown')

# The Fisher Method
cl=docclass.fisherclassifier(docclass.getwords)
docclass.sampletrain(cl)
cl.classify('quick rabbit')
cl.classify('quick money')
cl.setminimum('bad',0.8)
cl.classify('quick money')
cl.setminimum('good',0.4)
cl.classify('quick money')

```