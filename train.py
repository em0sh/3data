###########################
# Model Training
##########################

import numpy as np

# TODO:
	# Stochastic Gradient Descent
		# Cost Function: C(w,b) = (1/2n) SUM(||y(x)-a||^2, x)
		# Activation: w1*x1 + w2*x2 + w3*x3 ... = y1
	# Visualize training (Chart, bars, etc)
	# Back propogation (Training)

class Train:
	def norm(m, val):
		# Normalize according to specified 'm'ode:
			# s -> sigmoid
			# r -> rectified linear 
		# Sigmoid
		if m == 's':
			return 1 / (1 + np.exp(-val))

		# RELU
		elif m == 'r':
			val = abs(val)
			if val > 2.5:
				return 2.5
			else:
				return val
	def grad(n):
		# Perform stochastic gradient
			# n = instantiated net class
			# initialize l1
				# Set l1 equal to existing values
			# Perform operations on l1o
		pass

	def feed(n):
		# Traverse network, summing activations and weights
		for f, g in enumerate(n.l[1:]):
		# n.l represents the layer array
			# Add one to match enumerate to actual slicing
			f += 1
			print('f is {} l[f] {}'.format(f, n.l[f]))
			for y, z in enumerate(n.l[f]):
			# n.l[f] represents the individual element within the n.l array - sum this
				# Step through the weight array and sum
				print('  start of nested loops within parent l array')
				print('  count of individual element within array is : {}'.format(y))
				print('  inside of individual element of array ')
				for i, j in enumerate(n.w[f-1]):
					print('    count of nested weight loop is is : {}'.format(i))
					print('    weight array being used: {}'.format(n.w[i]))
					for k, l in enumerate(n.w[f-1]):
						print('      within most nested loop ')
						print('      nested weight array to add: {}'.format(l))
						for m, o in enumerate(n.w[f-1][k]):
							n.l[f][y] += n.w[f-1][k][m]
							print('        {} added to array element in l, which is now {}'.format(n.w[f-1][k][m],n.l[f][y]))

	def backProp(n):
		# Propogate error backwards through network
		pass


	def upWeights(n):
		# Update the weights based off of the backpropogation
		pass
