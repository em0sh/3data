################################
# Neural Network Class Instantiation
################################

# TODO:
<<<<<<< HEAD
    # Gradient descent
=======
	# Instantiate class with properties
>>>>>>> 5bec607604511dc1a82d33d0429dcf70230834be

# Standard Libraries
import random, copy
import numpy as np

# Modules
import train


class Net:
<<<<<<< HEAD
    def __init__(self, layers, batchSize, eth):
        self.lay = layers
        self.batchSize = batchSize
        self.eth = eth

        # Calculated Attributes
        self.laySize = len(self.lay)
    
    # Put function: generate the numbers to insert into the hidden layer array
    def put():
        # todo: take an argument that determines which method to initialize numbers
        return random.uniform(0,2)

    def initialize(self):
        # Initialize the network arrays
        # Create the network layers, l for manipulation
        self.l = []
        self.w = []
        self.a = []

        for j, i in enumerate(self.lay):
            # Insert zeroes for first and last layer of activations
            # Insert initial values for weights
            if j == 0:
                self.l.append([0 for z in range(self.lay[j])])
                self.w.append([[Net.put() for y in range(i)] for z in range(self.lay[j+1])])

            if j == self.laySize - 1:
                self.l.append([0 for z in range(self.lay[j])])

            # Insert initialized number into array for hidden layers
            self.l.append([Net.put() for z in range(self.lay[j])])

    def grad(self):
        # todo: Find a faster alternative to deepcopy
        self.l1 = copy.deepcopy(self.l)
        self.l1[0][0] = 1

        # TODO: Access to individual elements within activation array shown, continue grad
        for i, z in enumerate(self.l1):
            for j, y in enumerate(z):
                self.l1[i][j] = 1
        print(self.w[0]) 
=======
	def __init__(self, layers, batchSize, eta):
		self.lay = layers
		self.batchSize = batchSize
		self.eta = eta

		# Calculated Attributes
		self.laySize = len(self.lay)
	
	def put():
	# Put function: generate the numbers to insert into the hidden layer array
		return random.uniform(-1., 1.)

	def initialize(self):
		# Initialize the network arrays
		# Create the network layers
		# Activations
		self.l = []
		self.ll = []
		# Weights
		self.w = []
		# Biases
		self.b = []


		for j, i in enumerate(self.lay):
		# Insert zeroes for first and last layer, initialize hidden layers
			if j == 0:
				# First layer in network: should represent normalized input from image
				self.l.append([0. for z in range(self.lay[j])])
				self.ll.append([0. for z in range(self.lay[j])])
				continue

			if j == self.laySize-1:
				self.l.append([0. for z in range(self.lay[j])])
				self.ll.append([0. for z in range(self.lay[j])])
				self.w.append([[Net.put() for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				self.b.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				continue
			# Insert initialized number into array for hidden layers
			self.l.append([0. for z in range(self.lay[j])])
			self.ll.append([0. for z in range(self.lay[j])])
			self.w.append([[Net.put() for x in range(self.lay[j-1])] for y in range(self.lay[j])])
			self.b.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
>>>>>>> 5bec607604511dc1a82d33d0429dcf70230834be
