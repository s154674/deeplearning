import matplotlib.pyplot as plt
import math
import json

def handle(key):
    if 'bt' in key and 'nr' in key:
        return 0
    if 'bt' in key and 'r0' in key:  
        return 1
    if 'bt' in key and 'r1' in key:  
        return 2
    if 'bt' in key and 'r2' in key:  
        return 3
    if 'bf' in key and 'nr' in key:
        return 4
    if 'bf' in key and 'r0' in key:  
        return 5
    if 'bf' in key and 'r1' in key:  
        return 6
    if 'bf' in key and 'r2' in key:  
        return 7


names = ['bg_off', 'bg_on', 'bg_off_seed']
plt.figure(figsize=(10, 5))

with open('final_results_dict2.txt', 'r') as f:
  dict = f.read()
  dict = json.loads(dict)

new_dct = {}
for i, key in enumerate(dict.keys()):
    find = key.find('_checkpoint')

    label = key[16:find]
    print(label)
    string = []
    if 'ppo' in key:
        string.append('p_')
    if 'impala' in key:
        string.append('i_')
    if '_no_re_' in key:
        string.append('nr_')
    if 're_0' in key:
        string.append('r0_')
    if 're_1' in key:
        string.append('r1_')
    if 're_2' in key:
        string.append('r2_')
    if 'bg_true' in key:
        string.append('bt_')
    if 'bg_false' in key:
        string.append('bf_')
    if '75_check' in key:
        string.append('75')
    if '200_check' in key:
        string.append('200')
    if '500_check' in key:
        string.append('500')

    try:
        new_dct["".join(string)].append(dict[key])
    except Exception:
        new_dct["".join(string)] = []
        new_dct["".join(string)].append(dict[key])

    'ppo_no_re_bg_true_75'

    'i_nr_bt_75'

    'p_r0_bf_75'

p_75 = [None]*8
p_200 = [None]*8
p_500 = [None]*8
i_75 = [None]*8
i_200 = [None]*8
i_500 = [None]*8

print(new_dct)

for k, v in new_dct.items():
    if k.startswith('p_') and k.endswith('75'):
        index = handle(k)
        p_75[index] = v
    if k.startswith('p_') and k.endswith('200'):
        index = handle(k)
        p_200[index] = v
    if k.startswith('p_') and k.endswith('500'):
        index = handle(k)
        p_500[index] = v
    if k.startswith('i_') and k.endswith('75'):
        index = handle(k)
        i_75[index] = v
    if k.startswith('i_') and k.endswith('200'):
        index = handle(k)
        i_200[index] = v
    if k.startswith('i_') and k.endswith('500'):
        index = handle(k)
        i_500[index] = v

list_of_plots = [p_75, p_200, p_500, i_75, i_200, i_500]
labels = ['nr','r0','r1','r2']
labels2 = ['No Resets', '1st Reset', '2nd Reset', '3rd Reset']
# print(list_of_plots)
xs = [75, 200, 500]

for i in list_of_plots:
    for ii in i:
        temp = 0
        for iii in ii:
            temp += iii
        ii.append(temp/5)


plt.subplot(2,2,2)
# plt.ylim(12.5,20)
for i in range(4):
    plt.plot(xs,[p_75[i][5],p_200[i][5],p_500[i][5]], label=labels2[i])
    # plt.fill_between(xs, [min(p_75[i][:4]), min(p_200[i][:4]), min(p_500[i][:4])], [max(p_75[i][:4]), max(p_200[i][:4]), max(p_500[i][:4])], alpha=0.1)
# plt.legend()
plt.title('Dynamic Backgrounds', fontsize=16)


plt.subplot(2,2,1)
# plt.ylim(12.5,20)
for i in range(4):
    plt.plot(xs,[p_75[i+4][5],p_200[i+4][5],p_500[i+4][5]], label=labels2[i])
    # plt.fill_between(xs, [min(p_75[i+4][:4]), min(p_200[i+4][:4]), min(p_500[i+4][:4])], [max(p_75[i+4][:4]), max(p_200[i+4][:4]), max(p_500[i+4][:4])], alpha=0.1)
# plt.legend()

plt.title('Static Background', fontsize=16)
plt.ylabel('Original PPO', fontsize=16)

plt.subplot(2,2,4)
# plt.ylim(12.5,20)
for i in range(4):
    plt.plot(xs,[i_75[i][5],i_200[i][5],i_500[i][5]], label=labels2[i])
    # plt.fill_between(xs, [min(i_75[i][:4]), min(i_200[i][:4]), min(i_500[i][:4])], [max(i_75[i][:4]), max(i_200[i][:4]), max(i_500[i][:4])], alpha=0.1)
# plt.legend()
plt.xlabel('# Levels available when training', fontsize=16)


plt.subplot(2,2,3)
# plt.ylim(12.5,20)
for i in range(4):
    plt.plot(xs,[i_75[i+4][5],i_200[i+4][5],i_500[i+4][5]], label=labels2[i])
    # plt.fill_between(xs, [min(i_75[i+4][:4]), min(i_200[i+4][:4]), min(i_500[i+4][:4])], [max(i_75[i+4][:4]), max(i_200[i+4][:4]), max(i_500[i+4][:4])], alpha=0.1)
plt.ylabel('IMPALA Inspired', fontsize=16)
plt.xlabel('# Levels available when training', fontsize=16)
plt.show()

# print(new_dct)