import net, train, load
import numpy as np

# Test modules
from random import shuffle, randint
import copy

run = True

# Network Parameters
# Array of layers & activations
netShape = [28*28, 30, 30, 10]
# Network container
ne = []
# List of integers used for SGD
nc = []

# Batch size for descent
bSize = 60000
# Learning rate
eta = 1

# Number of iterations to test
tst = int(bSize/60)



# Generate a list, counting to the length of the training array, and shuffle it for SGD
for i in range(len(load.nd)):
	nc.append(i)
shuffle(nc)


def compute(iteration):
	# Instantiate network object
	ne.append(net.Net(netShape, bSize, eta))

	# Iterate through and perform one loop of SGD based on the batch size
	# ct: variable to hold count of correct guesses
	ct = 0

	# List of epochs for SGD
	ep = 0

	# DIAG: nc
	for i in range(bSize):
		# Neural net training loop
		# Set randomized net count to initialize training data with to x

		x = nc[i]

		# Increment Epoch count by one
		ep += 1

		# Set first layer to input
		ne[iteration].l[0] = load.nd[x]

		# Feedforward
		train.feed(ne[iteration], True)
		

		# Pass network and answer to backprop
		train.backProp(ne[iteration], load.genLabel(x))

		train.SGD(ne[iteration])

	shuffle(nc)

	for i in range(tst):
		if bSize < tst:
			print('breaking test count')
			break
		z = nc[i]

		ne[iteration].l[0] = load.nd[z]
		guess = np.argmax(train.feed(ne[iteration], False))
		answer = np.argmax(load.genLabel(z))

		if answer == guess:
			ct += 1
	
	print('correct %: {}, eta: {}'.format(ct/tst, eta))
		


	

t = 0

while run == True:
	eta -= .025
	compute(t)

	# For downwards counts
	if eta <= 0:
		run = False
	
	t += 1

		




# TODO: Cycle through test samples to gauge accuracy
# TODO: Save network for use later
# TODO: Open existing network to test


# TODO: Stochastic Gradient Descent (Inside Train module)
	# First weight layer is not updating for some reason - is this right? Check this later
