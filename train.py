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

		return(1 / (1 + np.exp(-val)))

	# RELU
	elif m == 'r':
		if val > 1.:
			return 1.
		if val < -1.:
			return -1.
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

def sigPrime(v):
	return norm('s', v)*(1-norm('s', v))

def QCF(n, nll, a):
	# Quadratic Cost Function: Calculate error against target values
	
	# Actual Error = (1/2) * [ Prediction - Actual ] ^ 2
	for ind in range(len(nll)):
		nll[ind] = .5 * (n[ind] - a[ind])**2
	return(nll)

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
				# Apply sigmoid at the layer level, f, y
				n.l[f][y] += n.w[f-1][y][m]*n.l[f-1][m]+n.b[f-1][y][m]

			# a = sig(z) -> this is applying the sigmoid after summing all of the elements above
			#print('n.l[f][y] -> n.l[{}][{}] = {}'.format(f, y, n.l[f][y]))
			#print('sigmoid of this is {}'.format(norm('s', n.l[f][y])))
			n.l[f][y] = norm('s', n.l[f][y])
			#print('new n.l value is {}'.format(n.l[f][y]))
			#input('')

def backProp(n, a):
	# Propogate error backwards through network
	# n.ww, n.bb contains the error, or delta for computation in stochastic gradient descent
	
	# Invoke QCF and set error for last layer
	n.ll[-1] = QCF(n.l[-1], n.ll[-1], a)



	for f, g in reversed(list(enumerate(n.ww))):
		# Statement above this loop handles last layer, the 1: indicing loops after this layer
		# DIAG:
		n.track.append(f)

		# DIAG:
		#print('backprop f = {}'.format(f))
		#input('')
		for y, z in enumerate(n.ww[f]):

			for m, o in enumerate(n.ww[f][y]):
				# Progress through ww and perform backprop calcs on m element of [f][y] array
				n.ww[f][y][m] += n.w[f][y][m]*n.ll[f+1][y]*sigPrime(n.l[f+1][y])
				n.bb[f][y][m] += sigPrime(n.b[f][y][m])


def SGD(n):
	# Sum batched deltas
	# This loop taken from backProp above
	for f, g in reversed(list(enumerate(n.w))):

		for y, z in enumerate(n.w[f]):
			for m, o in enumerate(n.w[f][y]):
				# Progress through ww and perform backprop calcs on m element of [f][y] array
				# CAUTION: SGD is dependent on batch size or n.bs, check on this in the future
				#print('w[{}][{}][{}] was {}, adjustment was {}'.format(f, y, m, n.w[f][y][m], n.eta / n.bs * n.ww[f][y][m]))
				n.w[f][y][m] -= n.eta / n.bs * n.ww[f][y][m]
				n.b[f][y][m] -= n.eta / n.bs * n.bb[f][y][m]
