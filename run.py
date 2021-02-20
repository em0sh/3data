import net, train, load
import numpy as np

# Test modules
from random import shuffle, randint
import copy

# Network Parameters
# Array of layers & activations
netShape = [28*28, 30, 10]
# Batch size for descent
bSize = 20000
# Learning rate
eta = .1



# Instantiate network object
ne = net.Net(netShape, bSize, eta)

def compute():

	# List of integers used for SGD
	nc = []

	# Generate a list, counting to the length of the training array, and shuffle it for SGD
	for i in range(len(load.nd)):
		nc.append(i)
	shuffle(nc)

	# List of epochs for SGD
	ep = 0

	# DIAG: nc
	for i in range(bSize):
		# Neural net training loop
		# Set randomized net count to initialize training data with to x
		# DIAG: x = i
		x = nc[i]
		# Increment Epoch count by one
		ep += 1

		# Set first layer to input
		ne.l[0] = load.nd[x]

		# Feedforward
		train.feed(ne, True)
		

		# Pass network and answer to backprop
		train.backProp(ne, load.genLabel(x))

		if ep % (bSize/10)  == 0:
			print('epoch {}'.format(ep))
			print(ne.l[-1])
			load.plot(list(ne.l[0]))
			print(load.genLabel(x))
		train.SGD(ne)



def test(z):
	ne.l[0] = load.nd[z]

	load.plot(list(ne.l[0]))	

	an = train.feed(ne, False)
	print(an)
	print('net guess is: {}'.format(np.argmax(an)))
	print('answer is: {}'.format(np.argmax(load.genLabel(z))))


compute()

while True:
	z = int(input('int: '))
	test(z)



		




# TODO: Cycle through test samples to gauge accuracy
# TODO: Save network for use later
# TODO: Open existing network to test


# TODO: Stochastic Gradient Descent (Inside Train module)
	# First weight layer is not updating for some reason - is this right? Check this later
