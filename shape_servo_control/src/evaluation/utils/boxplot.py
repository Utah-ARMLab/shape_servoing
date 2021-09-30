import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})  # change font to size larger
# plt.rc('xtick', labelsize=10)

recording_path = "/home/baothach/shape_servo_data/evaluation/visualization/plot_data_1/refined"

# with open(os.path.join(recording_path, "inside", "result.pickle"), 'rb') as handle:
#     all_inside_data = pickle.load(handle) 
# with open(os.path.join(recording_path, "outside", "result.pickle"), 'rb') as handle:
#     all_outside_data = pickle.load(handle) 

with open(os.path.join(recording_path, "result.pickle"), 'rb') as handle:
    all_data = pickle.load(handle) 


prim_name = "box"  #"cylinder"
# inside_data = all_inside_data[prim_name]
# outside_data = all_outside_data[prim_name]

# fig, ax = plt.subplots()
# x = ['1k Pa', '5k Pa', '10k Pa']
# all_data = [all_inside_data, all_outside_data]
# all_inside_data["test"] = np.array(['1k Pa', '5k Pa', '10k Pa']).astype('float')

ax=sns.boxplot(x="obj name", y="chamfer",hue='Category',data=all_data, showfliers = False)

plt.title('Experiment Results Over Multiple Objects', fontsize=16)
plt.xlabel('Object Names',fontsize=16)
plt.ylabel('Chamfer Distance (m)', fontsize=16)

# ax=sns.boxplot(data=all_inside_data["hemis"])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels,title='',loc='upper center', bbox_to_anchor=(0.5, 1.11),ncol=5, fancybox=False, shadow=False)
# print(y)
# print(inside_data[:,1)
# print(outside_data)
plt.show()
# 


