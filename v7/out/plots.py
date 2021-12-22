import glob, os
import matplotlib.pyplot as plt
import re

buffer = 3006464

xs = []
ys = []
loss = []
labels = []

# 54 = maze
# 56 = re pol ppo
# 59 = ppo
# 61 = re pol impala
# 65 = impala

for i, infile in enumerate(glob.glob(os.path.join('*.out'))):
    observer_count = 0
    xs.append([])
    ys.append([])
    loss.append([])
    labels.append([])
    # print(infile)
    with open(infile, 'r') as f:
        data = f.readlines()
        print(len(data))
        for ii, line in enumerate(data):
            if line.startswith('Observation space'):
                observer_count += 1
                if observer_count == 1:
                    labels[i].append(f'Training {data[ii-1]}')
               
            if line.startswith('Step'):
                temp = line.split(': ')
                xs[i].append(int(temp[1].split('\t')[0])+(observer_count-1)*buffer)
                ys[i].append(float(temp[2].split('\t')[0]))
                loss[i].append(float(temp[3][:-2]))
        

# print(xs)
# labels = ['ppo 1', 'ppo 2', 'ppo 3', 'impala 1', 'impala 2', 'impala 3']

def plot_all():
    for i, _ in enumerate(xs):
        # print(labels[i], labels[i][0])
        # if not labels[i][0].endswith("with policy reset"):
        plt.plot(xs[i], ys[i], label=labels[i][0])

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
        
        plt.plot(x,y,label=labels[i][0][:-2])

    plt.xlabel('Averaged Timestep')
    plt.ylabel('Averaged Mean Reward')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    stuff()
    # plot_all()