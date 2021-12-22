import matplotlib.pyplot as plt
import math
import json

names = ['bg_off', 'bg_on', 'bg_off_seed']
plt.figure(figsize=(10, 5))

with open('final_result_dict.txt', 'r') as f:
  dict = f.read()
  dict = json.loads(dict)

for i, key in enumerate(dict.keys()):
    find = key.find('_checkpoint')

    label = key[16:find]

    j = i % 3
    plotid = int(math.floor(i/3))+1
    val = dict[key]
    
    if i % 3 == 0:
        values = []
    values.append(val)   
    if i % 3 == 2:
        print(plotid)
        plt.subplot(2,6,plotid)
        plt.title(label)
        plt.bar(names, values, color=['#599ad3', '#f9a65a', '#9e66ab'])



plt.suptitle('Generalisation Plotting')
plt.show()




