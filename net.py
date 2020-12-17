################################
# Neural Network Class Instantiation
################################

# TODO:
	# Instantiate class with properties
		# Layers, Mini Batch Size, Input, Output
	# Count layers in network
	# Forward propogation (Initialization)

# Standard Libraries
import random
import numpy as np

# Modules
import train


class Net:
	def __init__(self, layers, batchSize, eth):
		self.lay = layers
		self.batchSize = batchSize
		self.eth = eth

		# Calculated Attributes
		self.laySize = len(self.lay)
	
	def put():
	# Put function: generate the numbers to insert into the hidden layer array
		return random.uniform(0,2)

	def initialize(self):
		# Initialize the network arrays
		# Create the network layers
		# Activations
		self.l = []
		# Weights
		self.w = []

		for j, i in enumerate(self.lay):
		# Insert zeroes for first and last layer
			if j == 0:
				self.l.append([0. for z in range(self.lay[j])])
				# First layer of weights should be zero since this is the input layer
				continue
			if j == self.laySize-1:
				self.l.append([0. for z in range(self.lay[j])])
				self.w.append([[1. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				continue
			# Insert initialized number into array for hidden layers
			self.l.append([0. for z in range(self.lay[j])])
			self.w.append([[1. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
