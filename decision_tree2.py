# -*- coding: utf-8 -*-

#%%
import numpy as np

class decision_node(object):
    def __init__(self, feature, value, t_branch=None, f_branch=None):
        self.feature = feature
        self.value = value
        self.t_branch = t_branch
        self.f_branch = f_branch         

class prediction_node(object):
    def __init__(self, t_support, f_support):
        self.t_support = t_support
        self.f_support = f_support
    def majority_class(self):
        if len(self.t_support) >= len(self.f_support):
            return self.t_support
        else:
            return self.f_support




def create_majority_prediction_node(samples):
    n_trues = len([label for _, label in samples if label])
    n_falses = len(samples) - n_trues
    return prediction_node(n_trues, n_falses)
 
def print_tree(tree, indent=''):
    if isinstance(tree, prediction_node):
        # Print the decision and support 
        print("Decision: {} (Support T:{}, F:{})".format(tree.majority_class(), tree.t_support, tree.f_support))
    else:
        # Print the criteria
        print(tree.feature,':',tree.value,'?')
        # Print the branches
        print(indent+' T->',end='')
        print_tree(tree.t_branch,indent+' ')
        print(indent+' F->',end='')
        print_tree(tree.f_branch,indent+' ')

#%%
def uniqueCounts(samples):
    unique_ct = {}
    for clases in samples:
        for classNode in clases[0]:
            if classNode not in unique_ct:
                unique_ct[classNode] = 0
            unique_ct[classNode] += 1
    
    return unique_ct




def entropy(samples):
    """ Receives a list of samples and calculates the class label entropy """
    unique_labels = uniqueCounts(samples)
    proportions = [ ]	
    for i in unique_labels.values():
        proportions.append(float(i)/len(samples))

    entropy = sum(-p*np.log2(p) for p in proportions)
    return entropy




#%%
def partition(samples, feature, value):
    """ Partition samples using the indicated feature on the value. In particular:
        - Categorical features are partitioned using the feature == value criteria
        - Numerical features are partitioned using the feature <= value criteria
        Ignores the samples without this feature
        Return the partition (part_t,part_f) """
    true_tree, false_tree = list(), list()
    
    if isinstance(value, int) or isinstance(value, float):
        def split_function(val): return  val <= value
    elif isinstance(value, str):
        def split_function(val): return val == value
    else:
        print("TypeError : Unsupported value type.")

    for sample in samples:
        if feature in sample[0]:

            if split_function(sample[0][feature]):
                true_tree.append(sample)
            else:
                false_tree.append(sample)
    
    return (true_tree, false_tree)





def information_gain(samples, feature, value):
	"""
	"""
	t_tree, t_false = partition(samples, feature, value)
	lensub1, lensub2 = len(t_tree), len(t_false)
	#if the node is pure return 0 entropy
	if lensub1 == 0 or lensub2 == 0:
		return (0, t_tree, t_false)
	#else calculate the weighted entropy
	weighted_ent = (len(t_tree)*entropy(t_tree) +
	                len(t_false)*entropy(t_false)) / len(samples)
	return ((entropy(samples) - weighted_ent), t_tree, t_false)



def best_partition(samples, split_features):
    """ Search the feature and value of the partition with minimum impurity
        Return a tuple with the feature, value and minimum impurity """
    # feature = ''
    # value = None
    # impurity = 0.0
    info_gain_features = []


    for feature in split_features:
        for sample in samples:
            if feature in sample[0]:
                info_gain_features.append(
                    (information_gain(samples, feature,
                                      sample[0][feature]), feature, sample[0][feature])
                )
                
                break
    
    best_entropy_pair = min(info_gain_features, key=lambda t: t[0][0])
   
    return (best_entropy_pair)


def build_tree(samples, split_features):
    """ Receive the samples and candidate split features 
        Return a decision tree """
    
    split, best_feature, gredee_value  = best_partition(samples, split_features)
    gain = float(split[0])
    samples_true = split[1]
    samples_false = split[2]

    if len(split_features) is 0 or entropy(samples) is 0.00:
       return prediction_node(samples_true, samples_false)
    
    if gain > 0:
        split_features.remove(best_feature)
        branch_true = build_tree(samples_true, split_features)
        branch_false = build_tree(samples_false, split_features)
        return decision_node(best_feature, gredee_value, branch_true, branch_false)
    else:
        return prediction_node(samples_true, samples_false)


def classify(observation : dict, tree):
    """ Return a prediction_node """
    observationKeys = list(observation.keys())
    observation_tuple = list(observation.items())[0]
    if isinstance(tree, prediction_node):
        return tree.majority_class()

    if tree.feature not in observationKeys:
        s_true = classify(observation, tree.t_branch)
        s_false = classify(observation, tree.f_branch)
        return prediction_node(s_true, s_false).majority_class()
    
    if isinstance(observation_tuple[1], int) or isinstance(observation_tuple[1], float):
        def defineVaule(): return observation_tuple[1] <= tree.value
    elif isinstance(observation_tuple[1], str):
        def defineVaule(): return observation_tuple[1] == tree.value
    else:
        print("TypeError : Unsupported value type.")

    if defineVaule():
        return classify(observation, tree.t_branch)
    else:
        return classify(observation, tree.f_branch)
interviewee = [
    ({'age': 52, 'english': 'elementary', 'experience': 'yes', 'degree': 'yes'}, True),
    ({'english': 'elementary', 'experience': 'no', 'degree': 'yes'}, False),
    ({'age': 45, 'english': 'advanced', 'experience': 'yes', 'degree': 'no'}, True),
    ({'age': 32, 'english': 'intermediate',
      'experience': 'yes', 'degree': 'yes'}, True),
    ({'age': 62, 'experience': 'no', 'degree': 'yes'}, False),
    ({'age': 21, 'english': 'elementary', 'experience': 'no', 'degree': 'yes'}, False),
    ({'age': 55, 'english': 'intermediate',
      'experience': 'yes', 'degree': 'yes'}, True),
    ({'age': 19, 'english': 'advanced', 'experience': 'no', 'degree': 'no'}, True),
    ({'age': 43, 'english': 'elementary', 'experience': 'yes', 'degree': 'no'}, False),
    ({'age': 32, 'english': 'advanced', 'experience': 'yes', 'degree': 'yes'}, True),
    ({'age': 60, 'english': 'elementary', 'experience': 'yes', 'degree': 'yes'}, True),
    ({'age': 46, 'english': 'intermediate',
      'experience': 'no', 'degree': 'no'}, False),
    ({'age': 24, 'english': 'elementary', 'experience': 'no', 'degree': 'no'}, False),
    ({'age': 18, 'english': 'intermediate',
      'experience': 'no', 'degree': 'no'}, False),
    ({'age': 59, 'english': 'advanced', 'experience': 'yes', 'degree': 'yes'}, True),
    ({'age': 68, 'english': 'advanced', 'experience': 'no', 'degree': 'yes'}, False)
]


#Prueba de Entropía
print("entropy partition")
print(entropy(interviewee))

advanced_english = [
    ({'english': 'elementary'}, False),
    ({}, False),
    ({'english': 'intermediate'}, False),
    ({'english': 'advanced'}, True)
]

# Test best_partition
print("______________________________")
print("Test Best partition")
split, best_feature, gredee_value = best_partition(
    advanced_english, 
    ['age', 'english', 'experience', 'degree']
    )
print("Split entropy t-tue, t.-false test")
print(split)
print("best best_feature ates")
print(best_feature)
print("Best gredee_value Test")
print(best_feature)


#Tree test
print("______________________________")
print("build_tree Test")
testTree = build_tree(interviewee, ['age', 'english', 'experience', 'degree'])
print_tree(testTree,'--')

#Tree decision
print("______________________________")
print("test clasify tree")
observation = {'english': 'intermediate', 'experience': 'yes',
               'degree': 'yes'}
print(classify(observation, testTree))
