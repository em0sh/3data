# Imports
import numpy as np
import struct, os, time, pickle
from array import array

# Library specifically designed for handling NMIST IDX
import idx2numpy

imgsize = 28
imagefile = 'data/trainimage'
labelfile = 'data/trainlabel'
output = 'data/trainfile'

# Reading
ndarr = idx2numpy.convert_from_file(imagefile)

class Figure():
    # Class for figure manipulation

    def plot(f):
    # Figure plotting
        # string to build
        s = ''

        for i in range(len(f)):
            for j in range(len(f[i])):
                if f[i][j] < 61:
                    s += ' '
                elif f[i][j] > 60 and f[i][j] < 190:
                    s += '.'
                else:
                    s += 'x'
            #    s += str(f[i][j])
            print(str(s))
            s = ''

    def grayscale(f):

        for i in range(len(f))
            for j in range(len(f[i])):
                if f[i][j] < 61:
                    f[i][j] = 0
                elif f[i][j] > 60 and f[i][j] < 190:
                    f[i][j] = 128
                else:
                    f[i][j] = 255
        return(f)

# Copy to array to work with
gsImages = ndarr.copy()

def label():		
	l = []
	with open(labelfile, 'rb') as file:
		magic, size = struct.unpack(">II", file.read(8))
		if magic != 2049:
			raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
		l = array("B", file.read())		
	return l


# TODO: Comment this code
def getData():
	idata = []
	ldata = label()
	for i in range(len(gsImages)):
		idata.append(Figure.grayscale(gsImages[i]))
		#Figure.plot(idata[i])
		print(i)
	dataz = list(zip(idata, ldata)) 
	return(dataz)

dz = getData()
w = open(output, 'wb')
pickle.dump(dz, w)
w.close()
print('complete')
