# Imports
import numpy as np
import struct, os, time, pickle
from array import array

# Library specifically designed for handling IDX
import idx2numpy

imgsize = 28
# Maximum value of byte - used for normalization
bSize = 255.
trainimage = 'data/trainimage'
trainlabel = 'data/trainlabel'
testimage = 'data/testimage'
testlabel = 'data/testlabel'


# initialize layer array and label array
nd = []
ndt = []

# Retrieve image data
# -------------------------------------------------------------------------------------------

# ndarrc is the array of the entire training set - put into nd and normalize
ndarrc = idx2numpy.convert_from_file(trainimage)
for i in ndarrc:
	nd.append(np.concatenate(i*(1/bSize), axis=None))

# ndarrct is the same as above, but for the testing image data
ndarrct = idx2numpy.convert_from_file(testimage)
for i in ndarrct:
	ndt.append(np.concatenate(i*(1/bSize), axis=None))

# Retrieve label data
# -------------------------------------------------------------------------------------------
# Training data
with open(trainlabel, 'rb') as file:
	magic, size = struct.unpack(">II", file.read(8))
	if magic != 2049:
		raise ValueError('2049 was magic, not {}'.format(magic))
	l = array("B", file.read())

# Test data
with open(testlabel, 'rb') as file:
	magic, size = struct.unpack(">II", file.read(8))
	if magic != 2049:
		raise ValueError('2049 was magic, not {}'.format(magic))
	lt = array("B", file.read())
	
	
# This is ugly, need to rework these two
def genLabel(t):
	label = np.zeros(10)
	label[l[t]] = 1.
	return(label)

def genLabelt(t):
	label = np.zeros(10)
	label[lt[t]] = 1.
	return(label)



################ Plotting
def plot(f):
# Figure plotting
	# string to build
	s = ''

	k = 0
	for i in range(imgsize):
		for j in range(imgsize):
			if f[j + k*imgsize] < 61/bSize:
				s += ' '
			elif f[j + k*imgsize] > 60/bSize and f[i] < 190/bSize:
				s += '.'
			else:
				s += 'x'
		print(str(s))
		k += 1
		s = ''

def grayscale(f):
	for i in range(len(f)):
		for j in range(len(f[i])):
			if f[i][j] < 61/bSize:
				f[i][j] = 0
			elif f[i][j] > 60/bSize and f[i][j] < 19/bSize0:
				f[i][j] = 128/bSize
			else:
				f[i][j] = 255/bSize
	return(f)


