# Imports
import numpy as np
import struct, os, time, pickle
from array import array

# Library specifically designed for handling IDX
import idx2numpy

imgsize = 28
imagefile = 'data/trainimage'
labelfile = 'data/trainlabel'

# Reading
ndarrc = idx2numpy.convert_from_file(imagefile)

# Length of array
ndarrl = len(ndarrc)

# initialize layer array and label array
nd = []
label = np.zeros(10)

# TODO: Comment this code
for i in ndarrc:
	nd.append(np.concatenate(i, axis=None))

# Retrieve label data
with open(labelfile, 'rb') as file:
	magic, size = struct.unpack(">II", file.read(8))
	if magic != 2049:
		raise ValueError('2049 was magic, not {}'.format(magic))
	l = array("B", file.read())
	
def genLabel(t):
	label[l[t]] = 1.
	return(label)

################ Plotting
def plot(f):
# Figure plotting
	# string to build
	s = ''

	k = 0
	for i in range(imgsize):
		for j in range(imgsize):
			if f[j + k*imgsize] < 61:
				s += ' '
			elif f[j + k*imgsize] > 60 and f[i] < 190:
				s += '.'
			else:
				s += 'x'
		print(str(s))
		k += 1
		s = ''

def grayscale(f):
	for i in range(len(f)):
		for j in range(len(f[i])):
			if f[i][j] < 61:
				f[i][j] = 0
			elif f[i][j] > 60 and f[i][j] < 190:
				f[i][j] = 128
			else:
				f[i][j] = 255
	return(f)


