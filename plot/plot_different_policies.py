import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
folders=["TD330","DDPG40","TD3_LSTM40"]
Names=["TD3","DDPG","TD3_LSTM"]
seeds = ["seed_31","seed_32"]
base_path = "logdir"  # your path to USV_planner/logdir
colors=['#1f77b4', '#ff7f0e', '#7f7f7f','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
lines = ['-','-',':','-',':']
markers=['d','^','s','^','^']
data={}
for folder in folders:
    floder_path = os.path.join(base_path, folder)
    floder_path = floder_path + "/marine_simulation"
    seeds_data = []
    for seed in seeds:
        seed_path = os.path.join(floder_path, seed) + "/results"
        files = os.listdir(seed_path)
        for file in files:
            if file[5:12] == '1000000':
                subdata = file
                break
        success_rate = np.load(os.path.join(seed_path,subdata))[:,0]
        seeds_data.append(success_rate)

    start = np.zeros((2,1))
    seeds_data = np.array(seeds_data)
    seeds_data = np.concatenate((start,seeds_data),axis=1)
    mean_data = np.mean(seeds_data, axis=0)
    std_data = np.std(seeds_data, axis=0)
    mean = ndimage.uniform_filter(mean_data, size=3)
    std = ndimage.uniform_filter(std_data, size=3)
    ub = mean + std / 4.
    lb = mean - std / 4.

    data[folder] = {
        'mean': mean,
        'max': ub,
        'min': lb
    }

plt.figure(figsize=(11,6))

for i,folder in enumerate(folders):
    mean = data[folder]['mean']
    max_val = data[folder]['max']
    min_val = data[folder]['min']
    x_axis_head = np.linspace(1,1.75,4)
    x_axis_head[0]=0
    x_axis_tail = np.linspace(2,100,50)
    x_axis = np.concatenate((x_axis_head,x_axis_tail))
    plt.plot(x_axis,mean, label=f'{Names[i]}', linewidth=2,color=colors[i],ls=lines[i],marker=markers[i],zorder=4-i)

    plt.fill_between(x_axis, min_val, max_val, alpha=0.3,color=colors[i],ls=lines[i],zorder=4-i)

plt.xlabel('Training steps (10^4)',fontsize=16)
plt.ylabel('Success rate',fontsize=16)
plt.tick_params(labelsize=16)
plt.legend()
plt.grid(True)
# plt.savefig("Success rate curves for different policy.svg",dpi=1000,bbox_inches='tight')
# plt.savefig("Success rate curves for different policy.png",dpi=500,bbox_inches='tight')
plt.show()