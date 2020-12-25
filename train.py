###########################
# Model Training
##########################

import numpy as np

# TODO:
	# Stochastic Gradient Descent
		# Cost Function: C(w,b) = (1/2n) SUM(||y(x)-a||^2, x)
	# Back propogation (Training)
	# Visualize training (Chart, bars, etc)

def norm(m, val):
	# Normalize according to specified 'm'ode:
		# s -> sigmoid
		# r -> rectified linear 
	# Sigmoid
	if m == 's':
		# Normalize to the maximum value of the input (255)
		val /= 255

		return(1 / (1 + np.exp(-val)))

	# RELU
	elif m == 'r':
		val = abs(val)
		if val > 2.5:
			return 2.5
		else:
			return val
	
	elif m == 'l':
		return val

def grad(n):
	# Perform stochastic gradient
		# n = instantiated net class
		# initialize l1
			# Set l1 equal to existing values
		# Perform operations on l1o
	pass

def err(pred, act):
	# Calculate error against target values
	# Error = (1/2) * [ Prediction - Actual ] ^ 2
	e = .5 * (pred - act)**2
	return(e)

def feed(n):
	# Traverse network, summing activations and weights

	for f, g in enumerate(n.l[1:]):
	# n.l represents the layer array
		# Add one to match enumerate to actual slicing
		f += 1
		for y, z in enumerate(n.l[f]):
		# n.l[f] represents the individual element within the n.l array - sum this
			for m, o in enumerate(n.w[f-1][y]):
				# Step through the weight array and sum
				# This Activation = Sigmoid * ( W * X + B )
				n.l[f][y] += norm('s', n.w[f-1][y][m]*n.l[f-1][m]+n.b[f-1][y][m])


def backProp(n, a):
	# Propogate error backwards through network
	
	for f, g in reversed(list(enumerate(n.l[1:]))):
		f += 1
		for y, z in enumerate(n.l[f]):
			for m, o in enumerate(n.w[f-1][y]):
				if f == len(n.l[:-1]):
					# Calculate error in the last layer
					err(n.l[f][y], a[y])
				else:
					# Backprop Calculations for remaining layers here, working backwards
					#	from the last layer in the network
					pass
def upWeights(n):
	# Update the weights based off of the backpropogation
	pass
