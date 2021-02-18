###########################
# Model Training
##########################
import numpy as np
from copy import deepcopy


# DIAG:
import load

# TODO:
	# Stochastic Gradient Descent
		# Cost Function: C(w,b) = (1/2n) SUM(||y(x)-a||^2, x)
	# Back propogation (Training)
	# Visualize training (Chart, bars, etc)
# DIAG:
diag = 0

def norm(val):
	return(1.0 / (1.0 + np.exp(-val)))

def sigPrime(v):
	return norm(v)*(1-norm(v))

def QCF(a, ll, y, z):
	# Quadratic Cost Function: Calculate error against target values
	
	# Actual Error = (1/2) * [ Prediction - Actual ] ^ 2
	for ind in range(len(ll)):
		# This version computes the actual statistical error
		#ll[ind] = .5 * (l[ind] - y[ind])**2

		# error at ll = (a - y) * sigprime(z)
		ll[ind] = (a[ind] - y[ind]) * sigPrime(z[ind])

	return(ll)

def feed(ne, train):
	# Traverse network, summing activations and weights

	# Set network equal to input array if true (for training)
	# Otherwise, copy for estimation
	if train == True:
		n = ne
	else:
		n = deepcopy(ne)

	for f, g in enumerate(n.w):
		# n.l represents the layer array

		for y, z in enumerate(n.w[f]):
			# n.l[f] represents the individual element within the n.l array
			for m, o in enumerate(n.w[f][y]):
				# Step through the weight array and sum
				# Apply sigmoid at the layer level, f, y
				# For the n.z array, using m seems comunter-intuitive, but this is because of how
				#	the n.w array is structured. y becomes m for z to maintain correct
				#	iteration

				n.z[f+1][m] += n.w[f][y][m]*n.l[f][y]+n.b[f][y][m]
	
		for i in range(len(n.l[f+1])):
			# a = sig(z) -> this is applying the sigmoid after summing all of the elements above
			n.l[f+1][i] = norm(n.z[f+1][i])
	

	# Return test value if testing (not training)
	if train == False:
		return(['%.9f' % x for x in  n.l[-1]])
	

def backProp(n, ans):
	# Propogate error backwards through network
	# n.ww, n.bb contains the error, or delta for computation in stochastic gradient descent
	
	# Invoke QCF and set error for last layer
	n.ll[-1] = QCF(n.l[-1], n.ll[-1], ans, n.z[-1])


	for f, g in reversed(list(enumerate(n.w))):

		# Statement above this loop handles last layer, the 1: indicing loops after this layer

		# Should skip first layer (Interpretaion from equations, need to check)
		if f == 0:
			break

		for y, z in enumerate(n.w[f]):

			for m, o in enumerate(n.w[f][y]):

				# Progress through ww and perform backprop calcs on m element of [f][y] array
				# DIAG: Are y and m here used correctly in the indicing? (n.ll[f][m] was y prior)
				n.ll[f][m] += n.w[f][y][m]*n.ll[f+1][m]*sigPrime(n.z[f][m])


def SGD(n):
	# Sum batched deltas
	# This loop taken from backProp above

	for f, g in reversed(list(enumerate(n.w))):

		f += 1

		# Should skip first layer (Interpretaion from equations, need to check)
		if f == 0:
			break

		for y, z in enumerate(n.w[f-1]):
			for m, o in enumerate(n.w[f-1][y]):
				# Progress through ww and perform backprop calcs on m element of [f][y] array
				n.w[f-1][y][m] -= n.eta* n.l[f-1][y]*n.ll[f][m]
				n.b[f-1][y][m] -= n.eta* n.ll[f][m]

	# Zero out arrays
	for f, g in enumerate(n.z):
		for y, z in enumerate(n.z[f]):
			n.z[f][y] = 0.
			n.ll[f][y] = 0.
