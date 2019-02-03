import numpy as np
from collections import defaultdict
#decision_tree from scratch using the ID3 algorithm.
#Assuming the input data is in the form of a dictionary 
#                       Col1    Col2    Col3      
#[	 
#	 [C1R1, C2R1, C3R1, ........], 
# 	 [C1R2, C2R2, C3R2, ........],
#	 .
#           .
#	'[C1RN, C2RN, C3RN, ........]
#]
#The coln represents the columns and the dictionary contains the rows, with 
#keys as the index.
#The last column represents the labels, as we are dealing with a supervised problem.
#The example with which we'll test will take integer values, we can further scale it to dealing 
#with categorical values.
def divide_data(data, column, value):
	"""
	returns a dataset which is split.
	"""
	split_function = None 
	if isinstance(value, int) or isinstance(value, float):
		split_function = lambda row : row[column] >= value
	elif isinstance(value, str):
		split_function = lambda row : row[column] == value
	else:
		print ("TypeError : Unsupported value type.")
	subset1 = [row for row in data if split_function(row)]
	subset2 = [row for row in data if not split_function(row)]			
	return (subset1, subset2)

def unique_counts(data):
	unique_ct = {}
	for row in data:
		print(row[len(row) - 1])
		if row[len(row) - 1] not in unique_ct:
			unique_ct[row[len(row) - 1]] = 0
		unique_ct[row[len(row) - 1]] += 1
	return unique_ct		

def entropy(data):
	"""
	For caluculating the cross-entropy of the data, using the class proportions 
	Also known as Shannon's Entropy.
	:params: label data as `label`
	:returns: Entropy of all the classes as a list
	"""
	unique_labels = unique_counts(data)
	proportions = []
	for i in unique_labels.values():
		proportions.append(float(i)/len(data))

	entropy = sum(-p*np.log2(p) for p in proportions)
	return entropy	

def information_gain(data, column, cut_point):
	"""
	For calculating the goodness of a split. The difference of the entropy of parent and 
	the weighted entropy of children.
	:params:attribute_index, labels of the node t as `labels` and cut point as `cut_point`
	:returns: The net entropy of partition 
	"""
	subset1, subset2 = divide_data(data, column, cut_point) 
	lensub1, lensub2 = len(subset1), len(subset2)  
	#if the node is pure return 0 entropy
	if len(subset1) == 0 or len(subset2) == 0:
		return (0, subset1, subset2)     
	#else calculate the weighted entropy 	
	weighted_ent = (len(subset1)*entropy(subset1) + len(subset2)*entropy(subset2)) / len(data)  
	return ((entropy(data) - weighted_ent), subset1, subset2)	 		


class dtree():
	#Every node will have a label
	attribute = None 
	cut_point = None 
	left = None
	right = None
	result = None
	fitness_function = None

	def __init__(self, attribute = None, cut_point = None, left = None, right = None, result = None, fitness_function = 'information_gain'):
		self.attribute = attribute
		self.cut_point = cut_point
		self.left = left
		self.right = right
		self.result = result
		if fitness_function ==  'information_gain':
			self.fitness_function = information_gain
		else:
			print ("function allocation failed")	

	def build_tree(self, rows):
		if len(rows) == 0: return dtree()	

		best_gain = 0
		best_criteria = None
		best_sets = None
		for column in range(len(rows[0] ) - 1):
			#the values to try splitting on
			unique_vals = {}
			for row in rows:
				unique_vals[row[column]] = 1
			#Now try to split at this column using every unique value it contains 	
			for uv in  unique_vals.keys():
				gain, sub1, sub2 = self.fitness_function(rows, column, uv)	
				if gain > best_gain:
					best_gain = gain
					best_criteria = (column, uv)
					best_sets = (sub1, sub2)
		#Now at this point, in the first call our decision tree has experienced its first split 
		#Now proceed only if best gain is worthful
		#if the best gain is 0, it simply means the leaf is pure.
		if best_gain > 0:
			left = self.build_tree(best_sets[0])
			right = self.build_tree(best_sets[1])		
			return dtree(attribute = best_criteria[0], cut_point = best_criteria[1], 
				left = left, right = right)
		else:
			return dtree(result = unique_counts(rows))	


	def print_tree(self, tree, indent = ' '):
		if tree.result != None:
			print (str(tree.result))
		else:
			print (str(tree.attribute) + ' : ' + str(tree.cut_point))	
			print (indent+'left->', end = ' ')
			self.print_tree(tree.left, indent = indent + '     ')
			print (indent+'right->', end = ' ')
			self.print_tree(tree.right, indent = indent + '     ')

	def classify(self, tree, row):
		if tree.result != None:
			return list(tree.result.keys())[0]
		else:
			branch = None
			if isinstance(tree.cut_point, int) or isinstance(tree.cut_point, float):
				if row[tree.attribute] >= tree.cut_point: branch = tree.left
				else : branch = tree.right 
			else:
				if row[tree.attribute] == tree.cut_point: branch = tree.left
				else : branch = tree.right
			return self.classify(branch, row)		

#COMPLETE THE BASIC STRUCTURE TOMORROW and they try hypothesis testing.			
