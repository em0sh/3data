###########################
# Model Training
##########################
import numpy as np

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

def QCF(l, ll, a):
	# Quadratic Cost Function: Calculate error against target values
	
	# Actual Error = (1/2) * [ Prediction - Actual ] ^ 2
	for ind in range(len(ll)):
		ll[ind] = .5 * (l[ind] - a[ind])**2
	return(ll)

def feed(n):
	print(n.w)
	print(len(n.w))
	print(len(n.w[0]))
	print(len(n.w[0][1]))
	input('')
	# Traverse network, summing activations and weights
	for f, g in enumerate(n.l[1:]):
		print('f starting as: {}'.format(f))
		input('')
	
	# n.l represents the layer array
		# Add one to match enumerate to actual slicing

		for y, z in enumerate(n.w[f]):
			print('f: {}'.format(f))
			print('elements in n.l array: {}'.format(len(n.l[f])))
			input('')
		# n.l[f] represents the individual element within the n.l array
			for m, o in enumerate(n.w[f][y]):
				print('elements in n.w array: {}'.format(len(n.w[f][y])))
				# Step through the weight array and sum
				# Apply sigmoid at the layer level, f, y

			
				print('f: {}, y: {}, m: {}'.format(f, y, m))
				print('n.z[f+1][y] += n.w[f][y][m]*n.l[f][y]+n.b[f][y][m]')
				print('{} += {} * {} + {}'.format(n.z[f+1][y], n.w[f][y][m], n.l[f][y], n.b[f][y][m]))
				n.z[f+1][y] += n.w[f][y][m]*n.l[f][y]+n.b[f][y][m]
				print('n.z[{}][{}] now: {}'.format(f+1, y, n.z[f][y]))
				#input('')

			# a = sig(z) -> this is applying the sigmoid after summing all of the elements above
			#print('n.l[f][y] -> n.l[{}][{}] = {}'.format(f, y, n.l[f][y]))
			#print('sigmoid of this is {}'.format(norm(n.l[f][y])))
			n.l[f][y] = norm(n.z[f][y])
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
		#n.track.append(f)

		# DIAG:
		#print('backprop f = {}'.format(f))
		#input('')
		for y, z in enumerate(n.ww[f]):

			for m, o in enumerate(n.ww[f][y]):
				# Progress through ww and perform backprop calcs on m element of [f][y] array
				n.ww[f][y][m] += n.w[f][y][m]*n.ll[f+1][y]*sigPrime(n.z[f+1][y])

				if n.ww[f][y][m] > 0.000000001 and diag == 1:
					print('n.w[f][y][m] = {}, n.ll[f+1][y] = {}, sigPrime(n.z[f+1][y]) = {}'.format(n.w[f][y][m], n.ll[f+1][y], sigPrime(n.z[f+1][y])))
					print('n.z[f+1][y] = {}'.format(n.z[f+1][y]))
					print(n.ww[f][y][m])
					print('n.ww[f][y][m] = n.ww[{}][{}][{}]'.format(f, y, m))
					input('')
				n.bb[f][y][m] += sigPrime(n.b[f][y][m])


def SGD(n):
	# Sum batched deltas
	# This loop taken from backProp above
	for f, g in reversed(list(enumerate(n.l[:1]))):
		for y, z in enumerate(n.w[f]):
			for m, o in enumerate(n.w[f][y]):
				# Progress through ww and perform backprop calcs on m element of [f][y] array
				# TODO: SGD is dependent on batch size or n.bs, check on this in the future
				#print('n[f][y][m] = n[{}][{}][{}]'.format(f, y, m))
				#print('n.l[f][y]*n.ww[f][y][m] = {} * {}'.format(n.l[f][y],n.ww[f][y][m]))
				n.w[f][y][m] -= n.eta / n.bs * n.l[f][y]*n.ww[f][y][m]
				n.b[f][y][m] -= n.eta / n.bs * n.bb[f][y][m]

