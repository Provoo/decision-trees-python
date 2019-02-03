"""
Neural net implemented through some really fun techiniques 
and tested upon MNSIT dataset.
"""
import numpy as np

import theano 
import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

#different types of neurons to choose from 
def linear(z): return z
def ReLU(z): return max(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

class FullyConnectedLayer(onject):

	def __init__(self, input_n, output_n, activation_fn = sigmoid, p_dropout = 0.0):
		"""
		Constructor for the fully connected layer. 
		params: input_n-The number of input neurons 
			  output_n-The number of outputs 
			  activation_fn-The activation function used, sigmoid by default 
			  p_dropout-the fraction of neurons selected for dropout 
		"""
		self.input_n = input_n
		self.output_n = output_n
		self.activation_fn = activation_fn
		self.p_dropout = p_dropout
		#initialise the weights and the biases 		
		self.w = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,
					scale=np.sqrt(1.0/output_n),
					size=(input_n, output_n)),
				dtype=theano.config.floatX),
			name='w',
			borrow=True
		)
		self.b = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,
					scale=1.0,
					size=(output_n,)),
				dtype=theano.config.floatX),
			name='b',
			borrow=True
		)
		#pack the params for easier use
		self.params=[self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		"""
		The PowerHouse. Takes the input, computes the output and 
		stores it as class variables to make it accessible by further layers 
		and for backpropogation too. 
		params: inpt-The input data 
			  inpt_dropout-Input data separated for dropout 
			  mini_batch_size-Number of training data used 
		"""
		self.inpt = inpt.reshape((mini_batch_size, self.input_n))
		self.output = self.activation_fn(
			(1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)

		self.y_out = T.argmax(self.output, axis=1)	
		self.inpt_dropout = dropout_layer(
			inpt_dropout.reshape((mini_batch_size, self.input_n)), 
			self.p_dropout)
		self.output_dropout = self.activation_fn(
			T.dot(self.inpt_dropout, self.w) + self.b)

	def accuracy(self, y):
		"""
		Return the accuracy the layer 
		params: y-the original output 
		"""
		return T.mean(T.eq(y, self.y_out))

class ConvolutionalLayer(object):

	def __init__(self, input_n, output_m, ):		 

		
class SoftmaxLayer(object):

	def __init__(self, input_n, output_n, p_dropout=0.0):		
		self.input_n = input_n
		self.output_n = output_n
		self.p_dropout = p_dropout
		self.w = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,
					scale=np.sqrt(1.0/self.output_n),
					size=(self.input_n, self.output_n)),
				dtype=theano.config.floatX
				)
			name='w',
			borrow=True
			)
		self.b = theano.shared(
			np.asarray(
				np.random.normal(
					loc=0.0,
					scale=np.sqrt(1.0/self.output_n),
					size=(output_n, )),
				dtype=theano.config.floatX
				)
			name='b',
			borrow=True
			)
		self.params = [self.w, self.b]
		
	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size, self.input_n))
		self.output = softmax((1-p_dropout)*T.dot(self.inpt, self.w) + self.b)
		self.y_out = T.argmax(self.output, axis=1)
		self.inpt_dropout = dropout_layer(
            			inpt_dropout.reshape((mini_batch_size, self.input_n)), self.p_dropout)
        		self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)	

        	def cost(self):
        		"""
        		Return the log likelihood cost(cost used by the softmax layer)
        		"""
        		return T.mean()
        	def accuracy(self, y):
        		"""
		Return the accuracy the layer 
		params: y-the original output 
		"""
        		return T.mean(T.eq(self.y_out, y))

		
class Network(object):
	def __init__(self, layers, mini_batch_size):
		self.layers = layers 
		self.mini_batch_size = mini_batch_size
		self.params = [param for layer in self.layers for param in layer.params]
		#initialise theano symbolic variables 
		self.x = T.matrix("x")
		self.y = T.ivector("y")
		init_layer = self.layers[0]
		init_layer.set_inpt(self.x, self.x, mini_batch_size)
		for j in xrange(1, len(self.layers)):
			prev_layer, cur_layer = self.layers[j-1], self.layers[j]
			cur_layer.set_inpt(
				prev_layer.output, 
				prev_layer.output_dropout, 
				self.mini_batch_size)
		self.output = self.layers[-1].output
		self.output_dropout = self.layers[-1].output_dropout

	def SGD(self, training_data, epochs, mini_batch_size, 
		eta, validation_data, test_data, lmda=0.0):
		"""
		The Powerhouse 
		"""		
		tr_len = len(training_data)
		te_len = len(test_data)
		va_len = len(validation_data)
		training_x, training_y = training_data
		validation_x, validation_y = validation_data
		test_x, test_y = test_data

		training_batch_num = size(training_data)/self.mini_batch_size
		validation_batch_num = size(validation_data)/self.mini_batch_size
		test_batch_num = size(test_data)/self.mini_batch_size

		l2_norm_term = sum([(layer.w**2).sum() for layer in self.layers])
		#The equation for the net cost, is dependent upin th cost 
		#used 
		cost = self.layers[-1].cost(self) + 0.5*l2_norm_term/training_batch_num
		#this is the backpropogation step
		grads = T.grad(cost, self.params)
		updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]

		i = T.lscalar()
		train_mb = theano.function(
			[i], cost, updates=updates,
			givens = {
				self.x:
				training_x[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
				self.y:
				training_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
			})

		validate_accuracy = theano.function(
			[i], self.layers[-1].accuracy(self.y),
			givens = {
				self.x:
				validation_x[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
				self.y:
				validation_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
			})

		test_accuracy = theano.function(
			[i], self.layers[-1].accuracy(self.y),
			givens = {
				self.x:
				test_x[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
				self.y:
				test_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]
			})
		self.test_predictions = theano.function(
			[i], self.layers[-1].y_out,
			givens = {
				self.x:
				test_x[i*mini_batch_size : (i+1)*mini_batch_size]
			})
		best_validation_accuracy = 0.0
		best_iteration 
		for epoch in epochs:
			for mini_batch_iteration in len(training_batch_num):
				iteration = training_batch_num*epoch + mini_batch_iteration
				if(iteration%1000 == 0):
					print("Training mini-batch number {0}".format(iteration))
				#do the updation 	
				cost_ij = train_mb(mini_batch_iteration)
				#now validate the process 
				if(iteration+1)%training_batch_num==0:
					validation_acc = np.mean(
						validate_accuracy(x) for x in range(validation_batch_num))
					print ("Validation accuracy until now, at {0} epoch is {1:.2%}".format(
						epoch, validation_acc))
					if validation_acc > best_validation_accuracy
						best_validation_accuracy = validation_acc
						best_iteration = iteration
						if test_data:
							test_acc = np.mean(
								test_accuracy(x) for x in range(test_batch_num))
						print ("Test accuracy corresponding to the best validation accuracy, at {0} epoch is {1:.2%}".format(
							epoch, test_acc))


def dropout_layer(layer, p_dropout):
	srng = shared_randomstreams.RandomStreams(
		np.random.RandomState(0).randint(999999))
	mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
	return layer*T.cast(mask, theano.config.floatX)						
