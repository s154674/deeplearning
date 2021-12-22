import glob, os
import matplotlib.pyplot as plt

xs = []
ys = []
loss = []

for i, infile in enumerate(glob.glob(os.path.join('*.out'))):
    xs.append([])
    ys.append([])
    loss.append([])
    # print(infile)
    with open(infile, 'r') as f:
        data = f.readlines()
        for line in data:
            if line.startswith('Step'):
                temp = line.split(': ')
                # print(temp)
                xs[i].append(int(temp[1].split('\t')[0]))
                ys[i].append(float(temp[2].split('\t')[0]))
                loss[i].append(float(temp[3][:-2]))

# print(xs)
labels = ['ppo 1', 'ppo 2', 'ppo 3', 'impala 1', 'impala 2', 'impala 3']

def plot_all():
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

def stuff():
    for i, _ in enumerate(xs):
        x, y = [], []
        tempx, tempy = xs[i], ys[i]
        while  tempx != []:
            x.append(sum(tempx[:10])/len(tempx[:10]))
            y.append(sum(tempy[:10])/len(tempy[:10]))
            tempx = tempx[10:]
            tempy = tempy[10:]
        
        plt.plot(x,y,label=labels[i])

    plt.xlabel('Averaged Timestep')
    plt.ylabel('Averaged Mean Reward')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    stuff()