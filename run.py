import net, train

n1 = net.Net([3, 2, 1], 10, .1)

n1.initialize()
train.Train.feed(n1)

print('n1.l[0] = {}'.format(n1.l[0]))
print('n1.l[1] = {}'.format(n1.l[1]))
print('n1.l[2] = {}'.format(n1.l[2]))
