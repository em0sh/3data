import net, train

n1 = net.Net([2, 2, 2], 10, .1)

n1.initialize()
n1.l[0][1] = 3

print('n1.l: {}'.format(n1.l))
print('n1.w: {}'.format(n1.w))
print('------- computation -------')

train.feed(n1)

print('n1.l: {}'.format(n1.l))
print('n1.w: {}'.format(n1.w))

print('n1.l[0] = {}'.format(n1.l[0]))
print('n1.l[1] = {}'.format(n1.l[1]))
print('n1.l[2] = {}'.format(n1.l[2]))
