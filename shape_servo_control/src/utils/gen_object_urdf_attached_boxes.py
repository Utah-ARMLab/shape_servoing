import os
import pickle

save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/multi_boxes_test"
object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_boxes_test"
shape_name = "box"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 0.7
attach_dist = 0.01



# with open(os.path.join(object_meshes_path, "primitive_dict_box.pickle"), 'rb') as handle:
#     data = pickle.load(handle)

for i in range(4,5):
    object_name = shape_name + "_" + str(i)

    # height = data[object_name]["height"]
    # width = data[object_name]["width"]
    # thickness = data[object_name]["thickness"]
    # youngs = round(data[object_name]["youngs"])

    height = 0.2
    width = 0.2
    thickness = 0.04
    youngs = 4000

    cur_urdf_path = save_urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = """    
    <robot name=\"""" + object_name + """\">
        <link name=\"""" + object_name + """\">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value=\"""" + str(density) + """\" />
                <youngs value=\"""" + str(youngs) + """\"/>
                <poissons value=\"""" + str(poissons) + """\" />
                <damping value="0.0" />
                <attachDistance value=\"""" + str(attach_dist) + """\" />
                <tetmesh filename=\"""" + object_meshes_path + """/""" + str(object_name) + """.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>

        <link name="fix_frame">
            <visual>
                <origin xyz=\"""" +str(-height*scale/2. - 10.0) + """ 0.0 0.0\"/>              
                <geometry>
                    <box size="0.005 0.15 """ + str(thickness*scale) + """\"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz=\"""" +str(-height*scale/2.) + """ 0.0 0.0\"/>              
                <geometry>
                    <box size="0.005 0.15 """ + str(thickness*scale) + """\"/>
                </geometry>
            </collision>
            <inertial>
                <mass value="500000"/>
                <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
            </inertial>
        </link>
        
        <joint name = "attach" type = "fixed">
            <origin xyz = "0.0 0.0 0.0" rpy = "0 0 0"/>
            <parent link =\"""" + object_name + """\"/>
            <child link = "fix_frame"/>
        </joint>  




    </robot>
    """
        f.write(urdf_str)
        f.close()

