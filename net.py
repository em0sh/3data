################################
# Neural Network Class Instantiation
################################

# TODO:
	# Instantiate class with properties

# Standard Libraries
import random
import numpy as np

# Modules
import train


class Net:
	def __init__(self, layers, batchSize, eta):
		self.lay = layers
		self.batchSize = batchSize
		self.eta = eta

		# Calculated Attributes
		self.laySize = len(self.lay)
		# DIAG: 
		self.track = []
	
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
		self.ww = []
		# Biases
		self.b = []
		self.bb = []


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
				self.ww.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				self.b.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				self.bb.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
				continue

			# Insert initialized number into array for hidden layers
			self.l.append([0. for z in range(self.lay[j])])
			self.ll.append([0. for z in range(self.lay[j])])
			self.w.append([[Net.put() for x in range(self.lay[j-1])] for y in range(self.lay[j])])
			self.ww.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
			self.b.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
			self.bb.append([[0. for x in range(self.lay[j-1])] for y in range(self.lay[j])])
