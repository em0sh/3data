import net, train, load

# Test modules
from random import randint

# Initial test 
# Instantiation of neural net class
	# net.Net([network array], batch size, learning rage)
n1 = net.Net([28*28, 25, 10], 10, .1)

# Generate network
	# TODO: Build this into the class initialization function instead
n1.initialize()

t = 0

# TODO: The final layer should change based on the first layer initializing to the image but it is not


# Set first and feed forward
n1.l[0] = load.nd[t]
train.feed(n1)
train.backProp(n1, load.genLabel(t))



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


''' print label and image
load.plot(list(n1.l[0]))
print(load.genLabel(t))
'''

