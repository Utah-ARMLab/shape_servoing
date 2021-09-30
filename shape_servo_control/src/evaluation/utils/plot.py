import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

recording_path = "/home/baothach/shape_servo_data/evaluation/chamfer_results/everything_raw"

with open(os.path.join(recording_path, "inside", "result.pickle"), 'rb') as handle:
    all_inside_data = pickle.load(handle) 
with open(os.path.join(recording_path, "outside", "result.pickle"), 'rb') as handle:
    all_outside_data = pickle.load(handle) 

prim_name = "box"  #"cylinder"
inside_data = all_inside_data[prim_name]
outside_data = all_outside_data[prim_name]

fig, ax = plt.subplots()
x = ['1k Pa', '5k Pa', '10k Pa']

y = inside_data[:,0]
line1 = ax.errorbar(x,y,yerr=inside_data[:,1], fmt='-o', capsize=5, label ="inside distribution")
y = outside_data[:,0]
line2 = ax.errorbar(x,y,yerr=outside_data[:,1], fmt='-^', capsize=5, label ="outside distribution")

ax.legend(handles =[line1, line2])
ax.set_title(prim_name)
ax.set_ylabel("Chamfer distance [m]")

print(inside_data[0])
# print(y)
# print(inside_data[:,1)
# print(outside_data)
plt.show()
# 


