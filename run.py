import net, train, load

n1 = net.Net([28*28, 30, 10], 10, .1)


n1.initialize()


n1.l[0] = load.nd[0]

print(n1.l[0])

train.feed(n1)

print(n1.l[2])
