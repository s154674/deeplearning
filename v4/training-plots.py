import glob, os
import matplotlib.pyplot as plt

xs = []
ys = []
loss = []

for i, infile in enumerate(glob.glob(os.path.join('*.out'))):
    xs.append([])
    ys.append([])
    loss.append([])
    print(infile)
    with open(infile, 'r') as f:
        data = f.readlines()
        for line in data:
            if line.startswith('Step'):
                temp = line.split(': ')
                print(temp)
                xs[i].append(int(temp[1].split('\t')[0]))
                ys[i].append(float(temp[2].split('\t')[0]))
                loss[i].append(float(temp[3][:-2]))

# print(xs)
labels = ['ppo startpilot 15 numaction','impala starpilot 15 numaction']


for i, _ in enumerate(xs):
    plt.plot(xs[i], ys[i], label=labels[i])

# # plt.plot(xs, ys)
# for x, y in zip(xs,ys):
#     plt.plot(xs[0], ys[0], label='1st')
#     plt.plot(x,y,f'')
#     plt.plot(xs[1], ys[1], label='2nd')
#     plt.plot(xs[2], ys[2], label='3rd')


plt.xlabel('Timestep')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()
