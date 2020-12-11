import numpy as np
import timeit, random

length = 1000000
al = [0]*length
aj = [0]*length

x = [random.randint(0,100) for k in range(length)]

def stand():
    for i, y in enumerate(al):
        i = i*x[y]

def nump():
    for j, y in enumerate(aj):
        j = np.multiply(j, x[y])

starttime = timeit.default_timer()
stand()
time1 = timeit.default_timer() - starttime
print(time1)

input('next ->')

starttime2 = timeit.default_timer()
nump()
time2 = timeit.default_timer() - starttime2
print(time2)
print('percentage time np took to complete: {}'.format(time2/time1))
