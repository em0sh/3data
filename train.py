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

				#print('f: {}, y: {}, m: {}'.format(f, y, m))
				#print('{} += {} * {} + {}'.format(n.z[f+1][m], n.w[f][y][m], n.l[f][y], n.b[f][y][m]))

				n.z[f+1][m] += n.w[f][y][m]*n.l[f][y]+n.b[f][y][m]

				#print('n.z[{}][{}] now: {}'.format(f+1, m, n.z[f+1][m]))

	
		for i in range(len(n.l[f+1])):
			# a = sig(z) -> this is applying the sigmoid after summing all of the elements above
			n.l[f+1][i] = norm(n.z[f+1][i])
	

	if train == False:
		print(['%.9f' % x for x in  n.l[-1]])
	

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
				#print('f = {}, y = {}, m = {}'.format(f, y, m))
				#print('n.ll[f][m] = {}'.format(n.ll[f][m]))
				#print('n.w[f][y][m] = {}'.format(n.w[f][y][m]))
				#print('n.ll[f+1][m] = {}'.format(n.ll[f+1][m]))
				#print('n.z[f][m] = {}'.format(n.z[f][m]))
				#print('sigPrime(n.z[f][m]) = {}'.format(sigPrime(n.z[f][m])))

				n.ll[f][y] += n.w[f][y][m]*n.ll[f+1][m]*sigPrime(n.z[f][m])

				#print('n.ll[f][m] = {}'.format(n.ll[f][m]))
			#input('')



				# DIAG: Realizing there should be one error container, ll, and that the difference
				#	between costs for weights in biases are in the equations and not variables
				#	such as n.ww and n.bb
				#n.bb[f-1][y][m] += sigPrime(n.b[f][y][m])

	# DIAG:
	#print('n.l[-1]: {}'.format(n.l[-1]))
	#print('n.ll[-1]: {}'.format(n.ll[-1]))
	#print('n.z[-1]: {}'.format(n.z[-1]))
	#input('')


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
				# TODO: SGD is dependent on batch size or n.bs, check on this in the future

				#print('-----------------------------')
				#print('f = {}, y = {}, m = {}'.format(f, y, m))
				#
				#print('n.w[{}][{}][{}] = {}'.format(f-1, y, m, n.w[f-1][y][m]))
				##print('n.eta/n.bs = {}'.format(n.eta/n.bs))
				#print('n.l[{}][{}] = {}'.format(f-1, y, n.l[f-1][y]))
				#print('n.ll[{}][{}] = {}'.format(f, m, n.ll[f][m]))
			
				n.w[f-1][y][m] -= n.eta* n.l[f-1][y]*n.ll[f][m]
				n.b[f-1][y][m] -= n.eta* n.ll[f][m]
				
				#print('calcuated:')
				#print('n.w[f-1][y][m] = {}'.format(n.w[f-1][y][m]))
				#print('n.b[f-1][y][m] = {}'.format(n.b[f-1][y][m]))

			#input('')


	# Zero out arrays
	for f, g in enumerate(n.z):
		for y, z in enumerate(n.z[f]):
			n.z[f][y] = 0.
			n.ll[f][y] = 0.
