#used for serializing and deserializing the objects 
import cPickle 
import gzip
#third party libraries 
import numpy as np


def load_data():
	f = gzip.open('dataset/data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	#returns a tuple of the data
	return (training_data, validation_data, test_data)

def load_data_wrapper():
	tr_d, va_d, te_d = load_data()
	training_input = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_output = [vectorize_output(num) for num in tr_d[1]]
	training_data = zip(training_input, training_output)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    	validation_data = zip(validation_inputs, va_d[1])
    	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    	test_data = zip(test_inputs, te_d[1])
    	return (training_data, validation_data, test_data)

def vectorize_output(num):
	"""
	vactorizes the output in the form of 10 binary bits 
	with the set bit denoting the number in the output

	params : num - the number 
	"""
	k = np.zeros((10, 1))
	k[num] = 1.0
	return k
