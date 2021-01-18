# TODO:
	# Implement Gradient Descent
		# Get total number of samples, n
		#? Average over batches?



import net, train, load

# Test modules
from random import randint

# Network Parameters
	# Array of layers & activations
netShape = [28*28, 30, 10]
	# Batch size for descent
batch = 10
	# Learning rate
eta = .1

# DIAG:
	# t is the example to choose
t = 0

# Instantiation of neural net class
n1 = net.Net(netShape, batch, eta)

# Set first and feed forward
n1.l[0] = load.nd[t]
train.feed(n1)
# Pass network and answer to backpropogation to populate delta matrices
train.backProp(n1, load.genLabel(t))













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

