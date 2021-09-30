import numpy as np
import pickle
import os
import pandas as pd

def filter_data(result_dir, prim_name, obj_type, inside, range_data):
    chamfer_data = []
    for i in range(range_data[0], range_data[1]):
        if inside:
            file_name = os.path.join(result_dir, obj_type, "inside", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, "outside", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
            # chamfer_data.extend(data)
            chamfer_data.extend([d for d in data if d <= 1])


    return chamfer_data


result_dir = "/home/baothach/shape_servo_data/evaluation/chamfer_results"
prim_name = "cylinder"
# obj_type = "cylinder_1k"
# prim_name = "box"
# obj_type = "box_5k"

# for obj_type in ['1k', '5k', '10k']:
#     print("==========")
#     print(f"{prim_name}_{obj_type}")
#     mean, std, data_length = get_mean_std(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=[0,10])
#     print(mean, std, data_length)

#     mean, std, data_length = get_mean_std(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=False, range_data=[0,10])
#     print(mean, std, data_length)


inside_data = {}
outside_data = {}

range_data = [0,10]
recording_path = "/home/baothach/shape_servo_data/evaluation/visualization/plot_data_1/raw"
for prim_name in ["cylinder", "box", "hemis"]:
    inside_data[prim_name] = []
    outside_data[prim_name] = []

    for obj_type in ['1k', '5k', '10k']:

        for inside in [True, False]:
            if inside:
                inside_data[prim_name].append(
                    np.array(filter_data(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=range_data)))
            else:
                outside_data[prim_name].append(
                    np.array(filter_data(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=False, range_data=range_data)))

    # inside_data[prim_name] = np.array(inside_data[prim_name])
    # outside_data[prim_name] = np.array(outside_data[prim_name])


# inside_data = pd.DataFrame.from_dict(inside_data)
# outside_data = pd.DataFrame.from_dict(outside_data)
# print(inside_data)
with open(os.path.join(recording_path, "inside", "result.pickle"), 'wb') as handle:
    pickle.dump(inside_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

with open(os.path.join(recording_path, "outside", "result.pickle"), 'wb') as handle:
    pickle.dump(outside_data, handle, protocol=pickle.HIGHEST_PROTOCOL)       


# with open(os.path.join(recording_path, "inside", "result.pickle"), 'rb') as handle:
#     inside_data = pickle.load(handle) 
# with open(os.path.join(recording_path, "outside", "result.pickle"), 'rb') as handle:
#     outside_data = pickle.load(handle) 


# datas = []
# object_names = []
# categories = []
# for prim_name in ["cylinder", "box", "hemis"]:
#     for obj_type in ['1k', '5k', '10k']:
#         chamfers = filter_data(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=True, range_data=range_data)
#         datas.extend(chamfers)
#         object_names.extend([f"{prim_name}_{obj_type}"]*len(chamfers))
#         categories.extend(["inside"]*len(chamfers))

#         chamfers = filter_data(result_dir, prim_name, f"{prim_name}_{obj_type}", inside=False, range_data=range_data)
#         datas.extend(chamfers)
#         object_names.extend([f"{prim_name}_{obj_type}"]*len(chamfers))
#         categories.extend(["outside"]*len(chamfers))


# df =  pd.DataFrame()
# df["chamfer"] = datas
# df["obj name"] = object_names
# df["category"] = categories
# print(df)

# with open(os.path.join(recording_path, "result.pickle"), 'wb') as handle:
#     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL) 

