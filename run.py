import net, train, load

# Test modules
from random import randint
# Initial test 
# Instantiation of neural net class
	# net.Net([network array], batch size, learning rage)
n1 = net.Net([28*28, 15, 10], 10, .1)

# Generate network
	# TODO: Build this into the class initialization function instead
n1.initialize()

t = 0

# TODO: The final layer should change based on the first layer initializing to the image but it is not
# Set first and feed forward
n1.l[0] = load.nd[t]*(1/255)
train.feed(n1)
train.backProp(n1, load.genLabel(t))
print(n1.l[-1])

''' print label and image
load.plot(list(n1.l[0]))
print(load.genLabel(t))
'''



# Second test
'''
n = []

for i in range(15):
	n.append(net.Net([28*28, 30, 10], 10, .1))



for i in n:
	i.initialize()
	t = randint(0, 50000)
	i.l[0] = load.nd[t]
	train.feed(i)

	print(i.l[-1])
'''
