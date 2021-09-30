from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

def setup_cam(gym, env, cam_width, cam_height, cam_pos, cam_target):
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height    
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    return cam_handle, cam_props


def get_particle_state(gym, sim):
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    return particle_state_tensor.numpy()[:, :3].astype('float32')  



def get_partial_point_cloud(gym, sim, env, cam_handle, cam_prop, min_z = 0.005, visualization=False):

    cam_width = cam_prop.width
    cam_height = cam_prop.height
    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    # print("Converting Depth images to point clouds. Have patience...")

    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 11] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] < -3:
                continue

            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            # print("p2:", p2)
            if p2[0, 2] > min_z:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        open3d.visualization.draw_geometries([pcd]) 

    return np.array(points).astype('float32')  


def default_sim_config(gym, args):
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.substeps = 4
    sim_params.dt = 1./60.
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.deterministic_mode = True    
    return gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params), sim_params

def default_dvrk_asset(gym, sim):
    # dvrk asset
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0005#0.0001

    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.max_angular_velocity = 40000.

    asset_root = "/home/baothach/icra2022_shape_servo_ws/src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    return gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)


def init_dvrk_joints(gym, env, dvrk_handle):
    dvrk_dof_states = gym.get_actor_dof_states(env, dvrk_handle, gymapi.STATE_NONE)
    dvrk_dof_states['pos'][8] = 1.5
    dvrk_dof_states['pos'][9] = 0.8
    gym.set_actor_dof_states(env, dvrk_handle, dvrk_dof_states, gymapi.STATE_POS)

def isaac_format_pose_to_PoseStamped(body_states):
    ros_pose = PoseStamped()
    ros_pose.header.frame_id = 'world'
    ros_pose.pose.position.x = body_states["pose"]["p"]["x"]
    ros_pose.pose.position.y = body_states["pose"]["p"]["y"]
    ros_pose.pose.position.z = body_states["pose"]["p"]["z"]
    ros_pose.pose.orientation.x = body_states["pose"]["r"]["x"]
    ros_pose.pose.orientation.y = body_states["pose"]["r"]["y"]
    ros_pose.pose.orientation.z = body_states["pose"]["r"]["z"]
    ros_pose.pose.orientation.w = body_states["pose"]["r"]["w"]
    return ros_pose

def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc


def isaac_format_pose_to_list(body_states):
    return [body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"],
        body_states["pose"]["r"]["x"], body_states["pose"]["r"]["y"], body_states["pose"]["r"]["z"], body_states["pose"]["r"]["w"]]

def fix_object_frame(object_world):
    object_world_fixed = deepcopy(object_world)
    object_size = [object_world.width, object_world.height, object_world.depth]
    quaternion =  [object_world_fixed.pose.orientation.x,object_world_fixed.pose.orientation.y,\
                                            object_world_fixed.pose.orientation.z,object_world_fixed.pose.orientation.w]  
    r = R.from_quat(quaternion)
    rot_mat = r.as_matrix()
    # print("**Before:", rot_mat)
    # x_axis = rot_mat[:3, 0]
    max_x_indices = []
    max_y_indices = []
    max_z_indices = []
    for i in range(3):
        column = [abs(value) for value in rot_mat[:, i]]
       
        if column.index(max(column)) == 0:
            max_x_indices.append(i)
        elif column.index(max(column)) == 1:
            max_y_indices.append(i)
        elif column.index(max(column)) == 2:
            max_z_indices.append(i)
    
    # print("indices: ", max_x_indices, max_y_indices, max_z_indices)
    if (not max_x_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_x_idx = max_z_indices[z_values.index(min(z_values))]
        max_y_idx = max_y_indices[0]
    elif (not max_y_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_y_idx = max_z_indices[z_values.index(min(z_values))]
        max_x_idx = max_x_indices[0]
    else:
        max_x_idx = max_x_indices[0]
        max_y_idx = max_y_indices[0]
        max_z_idx = max_z_indices[0]
        
    # print("indices:", max_x_idx, max_y_idx, max_z_idx)
    # x_values = [abs(x) for x in rot_mat[0, :]]
    # y_values = [abs(y) for y in rot_mat[1, :]]
    # z_values = [abs(z) for z in rot_mat[2, :]]
    # max_x_idx = x_values.index(max(x_values))
    # max_y_idx = y_values.index(max(y_values))
    # max_z_idx = z_values.index(max(z_values))

    fixed_x_axis = rot_mat[:, max_x_idx]
    fixed_y_axis = rot_mat[:, max_y_idx]
    fixed_z_axis = rot_mat[:, max_z_idx]
   
    fixed_rot_matrix = np.column_stack((fixed_x_axis, fixed_y_axis, fixed_z_axis))

    if (round(np.linalg.det(fixed_rot_matrix)) != 1):    # input matrices are not special orthogonal(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html)
        for i in range(3): 
            fixed_rot_matrix[i][0] = -fixed_rot_matrix[i][0]  # reverse x axis


    r = R.from_matrix(fixed_rot_matrix)
    fixed_quat = r.as_quat()
    print("**After:", fixed_rot_matrix)
    object_world_fixed.pose.orientation.x, object_world_fixed.pose.orientation.y, \
            object_world_fixed.pose.orientation.z, object_world_fixed.pose.orientation.w = fixed_quat
 

    r = R.from_quat(fixed_quat)
    print("matrix: ", r.as_matrix())

    
    object_world_fixed.width = object_size[max_x_idx]
    object_world_fixed.height = object_size[max_y_idx]
    object_world_fixed.depth = object_size[max_z_idx]

    return object_world_fixed