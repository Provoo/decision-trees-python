import numpy as np 

k = np.random.normal(
	loc = 0.0,
	scale = 1.0,
	size = (2, 4)
	)

k.reshape(4, 2)
print (k)