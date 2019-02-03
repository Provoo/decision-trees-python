"""
Driver program to test the neural network 
"""

import NeuralNetSGD
import mnsit_load

#get the training data
#training data is itself a tuple containing inputs and the outputs
training_data, validation_data, test_data= mnsit_load.load_data_wrapper()

net = NeuralNetSGD.NeuralNet([784, 30, 10], cost = NeuralNetSGD.CrossEntropyCost)
net.stochastic_gradient(training_data, 30, 10, 0.5, 5.0, 
	evaluation_data = validation_data, 
	monitor_evaluation_cost = True,
	monitor_evaluation_accuracy = True,
	monitor_training_cost = True, 
	monitor_training_accuracy = True)

