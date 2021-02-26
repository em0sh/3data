import net, train, load
import numpy as np

# Test modules
from random import shuffle, randint
import copy

# Network Parameters
# Array of layers & activations
netShape = [28*28, 30, 10]
# Batch size for descent
bSize = 60000
# Learning rate
eta = .1
# Number of iterations to test
tst = int(bSize/60)



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
	status = 0

	# DIAG: nc
	for i in range(bSize):
		# Neural net training loop
		# Set randomized net count to initialize training data with to x
		# DIAG: x = i
		x = nc[i]
		# Increment Epoch count by one
		status += 1

		# Set first layer to input
		ne.l[0] = load.nd[x]

		# Feedforward
		train.feed(ne, True)
		

		# Pass network and answer to backprop
		train.backProp(ne, load.genLabel(x))

		if status % (bSize/10)  == 0:
			print('status {}'.format(status))
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


for i in range(2):
	compute()
	print(f'epoch {i} complete')

# TESTING BLOCK - REMOVE BELOW TO NEXT TESTING BLOCK
nc = []

# Generate a list, counting to the length of the training array, and shuffle it for SGD
for i in range(len(load.nd)):
	nc.append(i)
shuffle(nc)

# Used for counting in test
ct = 0

for i in range(tst):
	if bSize < tst:
		print('breaking test count')
		break
	z = nc[i]

	ne.l[0] = load.nd[z]
	guess = np.argmax(train.feed(ne, False))
	answer = np.argmax(load.genLabel(z))

	if answer == guess:
		ct += 1

print('correct %: {}, eta: {}'.format(ct/tst, eta))
# TESTING BLOCK

while True:
	z = int(input('int: '))

	test(z)



		




# TODO: Cycle through test samples to gauge accuracy
# TODO: Save network for use later
# TODO: Open existing network to test


# TODO: Stochastic Gradient Descent (Inside Train module)
	# First weight layer is not updating for some reason - is this right? Check this later
