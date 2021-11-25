import glob, os
import matplotlib.pyplot as plt

xs = []
ys = []

for i, infile in enumerate(glob.glob(os.path.join('*.out'))):
    xs.append([])
    ys.append([])
    print(infile)
    with open(infile, 'r') as f:
        data = f.readlines()
        for line in data:
            if line.startswith('Step'):
                temp = line.split('Mean reward: ')
                xs[i].append(int(temp[0][6:-1]))
                ys[i].append(float(temp[1][:-1]))

# print(xs)


# plt.plot(xs, ys)
plt.plot(xs[0], ys[0], label='1st')
plt.plot(xs[1], ys[1], label='2nd')
plt.plot(xs[2], ys[2], label='3rd')
plt.xlabel('Timestep')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()
