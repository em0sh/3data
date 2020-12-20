import net, train, load

n1 = net.Net([28*28, 30, 10], 10, .1)


n1.initialize()

t = 59999

n1.l[0] = load.nd[t]

load.plot(list(n1.l[0]))
print(load.l[t])

train.feed(n1)

print(n1.l[2])
