"""
A simple program implementing neural nets from scratch.
Using MNSIT dataset for testing.
"""
import random
import json
import sys
# third party library
import numpy as np


class CrossEntropyCost(object):
	"""
	Methods for cross entropy function
	"""
	@staticmethod
	def fn(a, y):
		"""
		The main cross entropy function
		params : a - activations of of current neuron layer
			   y - expected outputs
		"""
		return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

	@staticmethod
	def delta(z, a, y):
		"""
		Output error for the cross entropy funciton
		"""
		return a - y


class QuadraticCost(object):
	"""
	Methods for the old-school quadratic cost function
	"""
	@staticmethod
	def fn(a, y):
		"""
		The main quadratic cost function
		params : a - activations of of current neuron layer
			   y - expected outputs
		"""
		return 0.5 * np.linalg.norm(a - y)**2

	@staticmethod
	def delta(z, a, y):
		"""
		The first back propogation equation also used this
		"""
		return (a - y) * sigmoid_prime(z)


class NeuralNet(object):
	"""
	Process explained -
		- Constructor gets called, random weights are assigned to the whole neural network.
		  Infact what the ultimate motive in NN is to train the weights and the bias only.
		- We do so using SGD in this code (The BEAST). So yeah, now we call stochastic_gradient
		  method.
	"""

	def __init__(self, sizes, cost=CrossEntropyCost, weight=None):
		"""
		Class constructor
		params : sizes is the array containing the neurons in each
			   cost - the cost function used while estimating the gradients
			             previously it was quadratic cost but now by default its the
			             cross entropy function.
		layer.
		"""
		self.num_layers = len(sizes)
		self.sizes = sizes
		# initialize the cost
		self.cost = cost
		# Initialise the weights
		if weight == None:
			self.default_weight_initialiser()
		else:
			self.large_weight_initializer()

	def default_weight_initialiser(self):
		"""
		Its used to initialise the weights and biases according to the modified method.
		Previously random variables from the standard normal distribution(mean 0 and std 1)
		were taken.
		But now its a bit modified to make the learning faster of the neural network.
		"""
		self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]
		self.weights = [np.random.randn(y, x) / np.sqrt(x)
				for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def large_weight_initializer(self):
		"""
		The previous method of initialising the weights.
		"""
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x)
		for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def feed_forward(self, a):
		"""
		This function is used to feedforward the neural net
		params : a is the list of activations
		"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		# a is the output
		return a

	def stochastic_gradient(self, training_data, epochs, mini_batch_size, eta,
		lmda=0.0,
		evaluation_data=None,
		monitor_evaluation_cost=False,
		monitor_evaluation_accuracy=False,
		monitor_training_cost=False,
		monitor_training_accuracy=False
		):
		"""
		This is the power of the neural network.
		What it does it learns the weights and the
		biases through the stochastic gradient method.

		params : training_data - the training data
			   epochs - the number of times we need to train the neural net
			   mini_batch_size - size of the randomly chosen batch
			   eta - the learning rate
			   lmda - the regularization constant, if set to zero there'll be no reg. effect
			   evaluation_data - data for testing
		"""
		if evaluation_data: n_eval = len(evaluation_data)
		n = len(training_data)
		evaluation_cost, evaluation_acc = [], []
		training_cost, training_acc = [], []
		# epoch is one net pass in the neural network
		for epoch in xrange(epochs):
			# shuffle the training data
			random.shuffle(training_data)
			mini_batches = [training_data[k: k + mini_batch_size]
			    for k in xrange(0, n, mini_batch_size)]
			#so instead of updating the gradient on the basis of 
			#whole dataset we supply a subset of it.    
			for mini_batch in mini_batches:
				self.update_params(mini_batch, eta, lmda, n)
			print "Epoch %s training complete" % epoch
			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, lmda, convert=True)
				print "Total cost on the evaluation data {}".format(cost)
				evaluation_cost.append(cost)
			if monitor_training_cost:
				cost = self.total_cost(training_data, lmda)
				print "Total cost on the training data {}".format(cost)
				training_cost.append(cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				print "Total accuracy on the evaluation data {} / {}".format(accuracy, n_eval)
				evaluation_acc.append(accuracy)
			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				print "Total accuracy on the training data {} / {}".format(accuracy, n)
				training_acc.append(accuracy)
			print

		return (
    evaluation_cost,
    evaluation_acc,
    training_cost,
     training_acc)

	def total_cost(self, data, lmda, convert=False):
		"""
		Used to evalute the total cost of the neural network.
		params : data - selfexplanatory
			   lmda - reg. constant
			   convert - convert the output for the validation data
		"""
		cost = 0.0
		for x, y in data:
			a = self.feed_forward(x)
			if convert: y = vectorize_output(y)
			cost += self.cost.fn(a, y)
		cost /= len(data)
		cost += 0.5 * (lmda / len(data)) * \
		               sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def accuracy(self, data, convert = False):
            	"""
		Returns the accuracy of the results
		We'll use np.argmax...
		convert will be true for training data else false 
            	"""      
            	acc_count = 0         
            	for x, y in data:
            		if convert : y = np.argmax(y)
            		a = np.argmax(self.feed_forward(x))
            		if a==y : acc_count += 1
            	return acc_count
            		
	def update_params(self, mini_batch, eta, lmda, n):
		"""
		Another core of the neural network
		Update the weights and the biases 
		params : mini_batch - the random picked up training data
			   eta - learning rate 
			   lmda - regularisation constant 
			   n - length of the training data 
		"""
		# the initial biases and the weights 
		ini_bias = [np.zeros(b.shape) for b in self.biases]
		ini_weights = [np.zeros(w.shape) for w in self.weights]
		# x and y are the features and the labels resp.
		# for each training example, do the work 
		for x, y in mini_batch:
			# biases after the back propogation
			back_bias, back_weights = self.backpropogate(x, y)
			# update the arrays 
			ini_bias = [ib + bb for ib, bb in zip(ini_bias, back_bias)]
			ini_weights = [iw + wb for iw, wb in zip(ini_weights, back_weights)]
		# now update the biases and the weights 	
		# theres no effect of regularisation on the biases 
		self.biases = [b - (eta / len(mini_batch)) * ib 
				for b, ib in zip(self.biases, ini_bias)]	
		# regularisation according to the L2 norm 						
		self.weights = [(1 - eta*(lmda/n))*w - (eta / len(mini_batch)) * wb 
				for w, wb in zip(self.weights, ini_weights)]	


	#################THESE FUNCTIONS ARE SHAMELESSLY COPIED#################

	def backpropogate(self, x, y):
	        	"""
		Algorithm - 
			-> Set the activation of the first layer as the input 
			-> Feed Forward 
			-> Compute the output error using the 1st back propogation eqution (using sigmoid prime) 
			-> Back propogate the error using the 2nd equation
			-> Get the gradient of the cost function using 3rd and the 4th equation 

	        	Return a tuple ``(nabla_b, nabla_w)`` representing the
	       	gradient for the cost function C_x.  ``nabla_b`` and
	       	 ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
	        	to ``self.biases`` and ``self.weights``."""

	        	nabla_b = [np.zeros(b.shape) for b in self.biases]
	        	nabla_w = [np.zeros(w.shape) for w in self.weights]
	        	###STEP-1###
	        	activation = x
	        	activations = [x] # list to store all the activations, layer by layer
	        	zs = [] # list to store all the z vectors, layer by layer
	        	###STEP-2###
	        	for b, w in zip(self.biases, self.weights):
	        		z = np.dot(w, activation) + b 
	        		zs.append(z)
	        		activation = sigmoid(z)
	        		activations.append(activation)

	        	###STEP-3###
	        	delta = self.cost.delta(zs[-1], activations[-1], y) * \
	        	    sigmoid_prime(zs[-1])
	        	###STEP-4###    
	        	nabla_b[-1] = delta
	        	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	        	# Note that the variable l in the loop below is used a little
	        	# differently to the notation in Chapter 2 of the book.  Here,
	        	# l = 1 means the last layer of neurons, l = 2 is the
	        	# second-last layer, and so on.  It's a renumbering of the
	        	# scheme in the book, used here to take advantage of the fact
	        	# that Python can use negative indices in lists.
	        	###STEP-5###
	        	for l in xrange(2, self.num_layers):
	        	    z = zs[-l]
	        	    sp = sigmoid_prime(z)
	        	    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	        	    nabla_b[-l] = delta
	        	    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	        	return (nabla_b, nabla_w)		 


	def save(self, filename):
        		"""Save the neural network to the file ``filename``."""
        		data = {"sizes": self.sizes,
                		"weights": [w.tolist() for w in self.weights],
                		"biases": [b.tolist() for b in self.biases],
                		"cost": str(self.cost.__name__)}
	        	f = open(filename, "w")
        		json.dump(data, f)
        		f.close()

	################## END END END OF COPIED FUNCTIONS#####################   
def load(filename):
    	"""Load a neural network from the file ``filename``.  Returns an
    	instance of Network.

    	"""
    	f = open(filename, "r")
    	data = json.load(f)
    	f.close()
    	cost = getattr(sys.modules[__name__], data["cost"])
    	net = Network(data["sizes"], cost=cost)
    	net.weights = [np.array(w) for w in data["weights"]]
    	net.biases = [np.array(b) for b in data["biases"]]
    	return net

def sigmoid(z):
	# z will be a vector 
	return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    	"""Derivative of the sigmoid function."""
    	return sigmoid(z)*(1-sigmoid(z))

def vectorize_output(num):
	"""
	vactorizes the output in the form of 10 binary bits 
	with the set bit denoting the number in the output

	params : num - the number 
	"""
	k = np.zeros((10, 1))
	k[num] = 1.0
	return k    	
