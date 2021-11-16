from numpy import ones
from numpy.random import random
 
# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.3)
def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3