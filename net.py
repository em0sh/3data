################################
# Neural Network Class Instantiation
################################

# Standard Libraries
import random
import numpy as np

# Modules
import train

# Notes
# n.x 	= initial definition of object
# n.xx 	= error


class Net:
	def __init__(self, layers, bSize, eta):
		self.lay = layers
		self.eta = eta

		# Calculated Attributes
		self.laySize = len(self.lay)
		# DIAG: 
		self.track = []

		# Initialize
		self.initialize()
		self.bs = bSize
	
	def put():
	# Put function: generate the numbers to insert into the hidden layer array
		# DIAG:
		return random.uniform(-1., 1.)
		#return .1


	def initialize(self):
		# Initialize the network arrays
		# Create the network layers
		# Activations
		self.l = []
		self.ll = []
		#	z is weighted input
		self.z = []
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
				self.z.append([0. for z in range(self.lay[j])])
				continue

			if j == self.laySize-1:
				self.l.append([0. for z in range(self.lay[j])])
				self.ll.append([0. for z in range(self.lay[j])])
				self.z.append([0. for z in range(self.lay[j])])
				self.w.append([[Net.put() for x in range(self.lay[j])] for y in range(self.lay[j-1])])
				self.b.append([[0. for x in range(self.lay[j])] for y in range(self.lay[j-1])])
				continue

			# Insert initialized number into array for hidden layers
			self.l.append([0. for z in range(self.lay[j])])
			self.ll.append([0. for z in range(self.lay[j])])
			self.z.append([0. for z in range(self.lay[j])])
			self.w.append([[Net.put() for x in range(self.lay[j])] for y in range(self.lay[j-1])])
			self.b.append([[0. for x in range(self.lay[j])] for y in range(self.lay[j-1])])





