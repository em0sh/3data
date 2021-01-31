# TODO:
	# Implement Gradient Descent
		# Get total number of samples, n
		#? Average over batches?
import net, train, load

# Test modules
from random import shuffle
import copy

# Network Parameters
# Array of layers & activations
netShape = [28*28, 30, 10]
# Batch size for descent
bSize = 100
# Learning rate
eta = .5


# Program Parameters
# List containing nets
ne = net.Net(netShape, bSize, eta)
# List of integers used for SGD
nc = []
# Number of training examples
m = 0

# Generate a list, counting to the length of the training array, and shuffle it for SGD
for i in range(len(load.nd)):
	nc.append(i)
shuffle(nc)

# DIAG: nc = [9]

timecounter = 0

def compute():
	# Iterate through and perform one loop of SGD based on the batch size

	# List of epochs for SGD
	ep = 0

	# DIAG:
	hold = copy.deepcopy(ne.w[0])

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
		train.feed(ne)
		# Pass network and answer to backprop
		train.backProp(ne, load.genLabel(x))

		# DIAG:
		if ep % 10  == 0:
			print('epoch {}'.format(ep))
			print(ne.l[1])
			load.plot(list(ne.l[0]))
			print(load.genLabel(x))
	train.SGD(ne)
	#print('network at start')
	#print(hold)
	print('network now')
	ne.w[0][-1][-1] = 5.
	print(ne.w[0])




compute()
		

# Iterate through this to get the input array and label



# TODO: Output progress
# TODO: Cycle through test samples to gauge accuracy
# TODO: Save network for use later
# TODO: Open existing network to test


# TODO: Stochastic Gradient Descent (Inside Train module)
	# First weight layer is not updating for some reason - is this right? Check this later











# DIAG:
#---------------------------------------------------------------------------------------------------
# show current layers for diagnostics
'''
z = 2
zz = z - 1
print('summary of layer l: {}'.format(z))
print('\nll')
print(n1.ll[z])
print(len(n1.ll[z]))

print('\nww')
print(n1.ww[zz])
print(len(n1.ww[zz]))

print('\nbb')
print(n1.bb[zz])
print(len(n1.bb[zz]))

# Print output layer of network
print("output layer of net")
print([ '%.1f' % el for el in n1.ll[-1]])

# Sum the outputs to diagnose changes in network
print('sum of output layer')
print('%.2f' % sum([i for i in n1.ll[-1]]))

print('f progressed in order: {}'.format(n1.track))
'''
# Print label and image
'''
load.plot(list(n1.l[0]))
print(load.genLabel(t))
'''
