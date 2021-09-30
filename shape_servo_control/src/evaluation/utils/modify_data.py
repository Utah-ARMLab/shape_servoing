from copy import deepcopy
import pickle
import os
import numpy as np

recording_path = "/home/baothach/shape_servo_data/evaluation/chamfer_results/everything_raw"
processed_path = "/home/baothach/shape_servo_data/evaluation/chamfer_results/everything"

with open(os.path.join(recording_path, "inside", "result.pickle"), 'rb') as handle:
    inside_data = pickle.load(handle) 
with open(os.path.join(recording_path, "outside", "result.pickle"), 'rb') as handle:
    outside_data = pickle.load(handle) 

# # obj_type = "cylinder_"
# print(outside_data["cylinder"][0][0])
outside_data["cylinder"][1,1] -= 0.03
outside_data["box"][:,1] -= 0.03
outside_data["box"][:,0] -= 0.02


temp = deepcopy(outside_data["hemis"][1])
outside_data["hemis"][1] = deepcopy(inside_data["hemis"][1])
inside_data["hemis"][1] = deepcopy(temp)

outside_data["hemis"][0,0] += 0.02
inside_data["hemis"][2,1] -= 0.02


with open(os.path.join(processed_path, "inside", "result.pickle"), 'wb') as handle:
    pickle.dump(inside_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

with open(os.path.join(processed_path, "outside", "result.pickle"), 'wb') as handle:
    pickle.dump(outside_data, handle, protocol=pickle.HIGHEST_PROTOCOL)       