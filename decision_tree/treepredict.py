# -*- coding:utf-8 -*-

# Read data from file
my_data = [line.strip().split('\t') for line in file('decision_tree_example.txt')]
# DataType transformation if necessary
data_cnt = len(my_data)
for r in range(data_cnt):
	my_data[r][3] = int(my_data[r][3])

# node in the tree
class decisionnode(object):
	def __init__(self, col = -1, value = None, results = None, tb = None, fb = None):
		self.col = col
		self.value = value
		self.results = results
		self.tb = tb
		self.fb = fb


# Divides a set on a specific column.
# Can hanle numeric or nominal values.
def divideset(rows, column, value):
	# Make a function that tells us if a row is in
	# the first group (true) or the second group (false)
	split_function = None
	if isinstance(value, int) or isinstance(value, float):
		split_function = lambda row: row[column] <= value
	else:
		split_function = lambda row: row[column] == value

	# Divide the rows into two sets and return them
	set1 = [row for row in rows if split_function(row)]
	set2 = [row for row in rows if not split_function(row)]
	return (set1, set2)


# Create counts of possible results
# (the last column of each row is the result)
def uniquecounts(rows):
	results = {}
	for row in rows:
		r = row[len(row) - 1]
		if r not in results:
			results[r] = 0
		results[r] += 1
	return results


# Gini Index (CART) of randomly partition of one set
# (the lower value, the better partition)
def giniimpurity(rows):
	total = len(rows)
	counts = uniquecounts(rows)
	imp = 0.0
	# Now calculate the Gini Index
	for j in counts:
		p = float(counts[j]) / total
		imp += p * (1 - p)
	return imp


# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results. Lower, better.
def entropy(rows):
	from math import log
	log2 = lambda x: log(x) / log(2)
	total = len(rows)
	counts = uniquecounts(rows)
	ent = 0.0
	# Now calculate the entropy
	for j in counts:
		p = float(counts[j]) / total
		ent = ent - p * log2(p)
	return ent


# Dealing with numerical outcomes (class)
def variance(rows):
	if len(rows) == 0:
		return 0
	data = [float(row[len(row)-1]) for row in rows]
	mean = sum(data) / len(data)
	variance = sum([(d - mean) ** 2 for d in data]) / len(data)
	return variance


# Build decision tree, default using entropy
def buildtree(rows, scoref = entropy):
	if len(rows) == 0:
		return decisionnode()
	current_score = scoref(rows)
	print current_score

	# Set up some variables to track the best criteria
	best_gain = 0.0
	best_criteria = None
	best_sets = None

	column_count = len(rows[0]) - 1
	for col in range(0, column_count):
		# Generate the list of different values in this column
		column_values = []
		for row in rows:
			column_values.append(row[col])
		# Try dividing the rows up for each value in this column
		for value in column_values:
			(set1, set2) = divideset(rows, col, value)
			# Information gain （ID3)
			p = float(len(set1)) / len(rows)
			gain = current_score - p * scoref(set1) - (1-p) * scoref(set2)
			if gain > best_gain and len(set1) > 0 and len(set2) > 0:
				best_gain = gain
				best_criteria = (col, value)
				best_sets = (set1, set2)

	# Create the subbranches
	if best_gain > 0:
		trueBranch = buildtree(best_sets[0])
		falseBranch = buildtree(best_sets[1])
		return decisionnode(col = best_criteria[0], value = best_criteria[1], 
							tb = trueBranch, fb = falseBranch)
	else:
		return decisionnode(results = uniquecounts(rows))


# Displaying the tree in plain text
def printtree(tree, indent = ''):
	# Is this a leaf node?
	if tree.results != None:
		print str(tree.results)
	else:
		# Print the criteria
		print str(tree.col) + ':' + str(tree.value) + '?'

		# Print the branches
		print indent + 'T->',
		printtree(tree.tb, indent + '  ')
		print indent + 'F->',
		printtree(tree.fb, indent + '  ')


# Get basic infos about the binary tree
def getwidth(tree):
	if tree.tb == None and tree.fb == None:
		return 1
	return getwidth(tree.tb) + getwidth(tree.fb)

def getdepth(tree):
	if tree.tb == None and tree.fb == None:
		return 0
	return max(getdepth(tree.tb), getdepth(tree.fb)) + 1

from PIL import Image, ImageDraw

# Draw whole binary tree
def drawtree(tree, jpeg = 'tree.jpg'):
	w = getwidth(tree) * 100
	h = getdepth(tree) * 100 + 120

	img = Image.new('RGB', (w, h), (255, 255, 255))
	draw = ImageDraw.Draw(img)

	centerPos = getwidth(tree.tb) * 100 - 50
	drawnode(draw, tree, centerPos, 20)
	img.save(jpeg, 'JPEG')

# Draw node of binary tree
def drawnode(draw, tree, x, y):
	if tree.results == None:
		# Get the width of each branch
		if tree.tb.fb != None and tree.fb.tb != None:
			w1 = getwidth(tree.tb.fb) * 100
			w2 = getwidth(tree.fb.tb) * 100
		else:
			w1 = 50
			w2 = 50

		# Determine the branch node position
		leftPos = x - w1
		rightPos = x + w2

		# Draw the condition string
		draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

		# Draw links to the branches
		draw.line((x, y, leftPos, y + 100), fill = (255, 0, 0))
		draw.line((x, y, rightPos, y + 100), fill = (255, 0, 0))

		# Draw the branch nodes
		drawnode(draw, tree.tb, leftPos, y + 100)
		drawnode(draw, tree.fb, rightPos, y + 100)
	else:
		txt = '\n'.join(['%s:%d ' % v for v in tree.results.items()])
		draw.text((x - 20, y), txt, (0, 0, 0))


# Classify unseen observation
def classify(observation, tree):
	if tree.results != None:
		return tree.results
	else:
		v = observation[tree.col]
		branch = None
		if isinstance(v, int) or isinstance(v, float):
			if v <= tree.value:
				branch = tree.tb
			else:
				branch = tree.fb
		else:
			if v == tree.value:
				branch = tree.tb
			else:
				branch = tree.fb
		return classify(observation, branch)


# Pruning the tree, post-prune
def prune(tree, mingain):
	# If the branches aren't leaves, then prune them
	if tree.tb.results == None:
		prune(tree.tb, mingain)
	if tree.fb.results == None:
		prune(tree.fb, mingain)

	# If both the subbranches are now leaves, see if they should merged
	if tree.tb.results != None and tree.fb.results != None:
		# Build a combined dataset
		tb, fb = [], []
		for v, c in tree.tb.results.items():
			tb += [[v]] * c 	# Construct class list
		for v, c in tree.fb.results.items():
			fb += [[v]] * c 	# Construct class list

		# Test the reduction in entropy
		delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)
		if delta < mingain:
			# Merge the branches
			tree.tb, tree.fb = None, None
			tree.results = uniquecounts(tb + fb)


# Dealing with missing data while predicting
def mdclassify(observation, tree):
	if tree.results != None:
		return tree.results
	else:
		v = observation[tree.col]
		if v == None:
			tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
			tcount = sum(tr.values())
			fcount = sum(fr.values())
			tw = float(tcount) / (tcount + fcount)	# weight
			fw = float(fcount) / (tcount + fcount)	# weight
			result = {}
			for k, v in tr.items():
				result[k] = v * tw
			for k, v in fr.items():
				result[k] = v * fw
			return result
		else:
			if isinstance(v, int) or isinstance(v, float):
				if v <= tree.value:
					branch = tree.tb
				else:
					branch = tree.fb
			else:
				if v == tree.value:
					branch = tree.tb
				else:
					branch = tree.fb
			return mdclassify(observation, branch)
