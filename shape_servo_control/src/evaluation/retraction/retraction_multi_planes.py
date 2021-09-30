#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import deepcopy
import rospy
import pickle
import timeit
import open3d

from geometry_msgs.msg import PoseStamped, Pose

from utils.isaac_utils import *
from utils.grasp_utils import GraspClient
from goal_plane import *

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl

sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
# from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
from architecture import DeformerNet
import torch
from PIL import Image


ROBOT_Z_OFFSET = 0.25




if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="dvrk",
        custom_parameters=[
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])


    # configure sim
    sim, sim_params = default_sim_config(gym, args)


    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)

    # create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()


    # load robot asset
    dvrk_asset = default_dvrk_asset()
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    


    # Load deformable object
    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    soft_asset_file = 'thin_tissue_layer_attached.urdf'
    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.035, 0.37-0.86, 0.081)  # 0.08
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)  
    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       

    # load kidney
    rigid_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    rigid_asset_file = "kidney_rigid.urdf"
    rigid_pose = gymapi.Transform()
    rigid_pose.p = gymapi.Vec3(0.00, 0.38-0.86, 0.03)
    rigid_pose.r = gymapi.Quat(0.0, 0.0, -0.707107, 0.707107)
    asset_options.thickness = 0.003 # 0.002
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, asset_options)


    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))
  

    # cache some common handles for later use
    envs = []
    dvrk_handles = []
    object_handles = []


    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)    
        dvrk_handles.append(dvrk_handle)    
        

        # add soft obj            
        soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)

        # add kidney
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, 'rigid', i, 0, segmentationId=11)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    
    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)    # set dof properties 


    # Viewer camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    
    # Camera for point cloud setup
    cam_handles = []
    cam_width = 256
    cam_height = 256 
    cam_positions = gymapi.Vec3(0.17, -0.62, 0.2)
    cam_targets = gymapi.Vec3(0.0, 0.40-0.86, 0.01)
    cam_handle, cam_prop = setup_cam(gym, envs[0], cam_width, cam_height, cam_pos, cam_target)
    # for i, env in enumerate(envs):
    #     cam_handles.append(cam_handle)


       

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('dvrk_shape_control')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
    rospy.logerr(f"Object type ... {args.obj_type}; inside: {args.inside}")  


    # Initilize robot's joints
    init_dvrk_joints(gym, envs[i], dvrk_handles[i])      


    # Set up DNN:
    device = torch.device("cuda")
    model = DeformerNet(normal_channel=False)
    # model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run1/epoch 150"))
    model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run2(on_ground)/epoch 128"))
    model.eval()


    # Some important paramters
    saved_plane_path = "/home/baothach/shape_servo_data/generalization/plane/computed_mani_points"
    data_recording_path = "/home/baothach/shape_servo_data/generalization/plane/results"   # success or nah


    all_done = False
    state = "home" 
    first_time = True
    save_intial_pc = True
    get_goal_pc = True
    dc_client = GraspClient()
    saved_plane_path = "/home/baothach/shape_servo_data/generalization/plane/computed_mani_points"
    data_recording_path = "/home/baothach/shape_servo_data/generalization/plane/results"   # success or nah
    execute_count = 0
    plane_count = 0
    num_new_goal = 0
    max_execute_count = 3
    max_plane_count = 100
    max_num_new_goal = 10    
    max_plane_time = 2 * 60 # 2 mins


    
    start_time = timeit.default_timer()   
    close_viewer = False
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)

        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            

            if frame_count == 10:
                rospy.logerr("**Current state: " + state + ", current plane count: " + str(plane_count))

                if first_time:
                    frame_state = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)                
                    frame_state['pose']['p']['z'] -= 0.05                
                    gym.set_actor_rigid_body_states(envs[i], object_handles[i], frame_state, gymapi.STATE_ALL) 
                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                    first_time = False



                # Get plane and computed manipulation point:
                with open(os.path.join(saved_plane_path, f"plane_{plane_count}.pickle"), 'rb') as handle:
                    plane_data = pickle.load(handle) 
                    constrain_plane = plane_data["plane"]
                    mani_point = plane_data["mani_point"]


                state = "generate preshape"
                
                frame_count = 0



        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            x, y, z = mani_point[0], mani_point[1], mani_point[2], 
            target_pose = [-x, -y, z-ROBOT_Z_OFFSET, 0, 0.707107, 0.707107, 0]



            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"                
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!") 

        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"
                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), g_2_pos)         
        
                anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
                start_vis_cam = True

                plane_start_time = timeit.default_timer()




        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
            print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 

            current_pc = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop)
  
            
            # num_points = current_pc.shape[0]         
            if save_initial_pc:
                initial_pc = deepcopy(current_pc)
                save_initial_pc = False
            
            if get_goal_pc:
                delta = 0.00
                goal_pc_numpy = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc)                         
                goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                get_goal_pc = False

            current_pc = torch.from_numpy(np.swapaxes(current_pc,0,1)).float()         

            with torch.no_grad():
                desired_position = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001)  
                # desired_position = model(intial_pc_tensor.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001) 

            print("from model:", desired_position)
          
            delta_x = desired_position[0]   
            delta_y = desired_position[1] 
            delta_z = desired_position[2] 

          

            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
            cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z
            # cartesian_pose.position.z = max(0.005- ROBOT_Z_OFFSET,current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z)
            dof_states = gym.get_actor_dof_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)['pos']

            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=dof_states)
            state = "move to goal"
            traj_index = 0

        if state == "move to goal":           
            # Does plan exist?
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "get shape servo plan"
            else:            
                # if frame_count % 10 == 0:
                #     current_pc = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop)()
                #     num_points = current_pc.shape[0]                    
                #     num_failed_points = len([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]]) 
                #     rospy.logwarn(f"percentage passed: {1-float(num_failed_points)/float(num_points)}")
                #     print("num, num failed:", num_points, num_failed_points)
                # frame_count += 1


                if num_new_goal >= max_num_new_goal \
                        or timeit.default_timer() - plane_start_time >= max_plane_time:

                    current_pc = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop)()
                    num_points = current_pc.shape[0]                    
                    num_failed_points = len([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]]) 

                    final_data = {"success": False, "% passed": 1-float(num_failed_points)/float(num_points)}
                    
                    with open(os.path.join(data_recording_path, f"plane_{plane_count}.pickle"), 'wb') as handle:
                        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)                       

                    rospy.logerr("FAIL: plane count " + str(plane_count))                             
                    plane_count += 1
                    state = "reset"




                dof_states = gym.get_actor_dof_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)['pos']
                plan_traj_with_gripper = [plan+[0.15,-0.15] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                
                
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                    traj_index += 1 

        
                if traj_index == len(plan_traj):
                    traj_index = 0  

                    
                    state = "get shape servo plan" 
                    execute_count += 1

                    if execute_count >= max_execute_count:
                        rospy.logwarn("Shift goal plane")
                        
                        delta += 0.02
                        new_goal = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc, check=True, delta=delta, current_pc=get_partial_point_cloud(i))
                        if new_goal is not None:
                            goal_pc_numpy = new_goal
                        
                        
                        num_new_goal += 1
                        print("num_new_goal:", num_new_goal)

                        if goal_pc_numpy is 'success':
                            final_data = {"success": True}
                            with open(os.path.join(data_recording_path, f"plane_{plane_count}.pickle"), 'wb') as handle:
                                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)                       

                            rospy.logerr("SUCCESS: plane count " + str(plane_count))                            
                            plane_count += 1
                            state = "reset"
                            
                    
                        else:
                            goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                            execute_count = 0
                            frame_count = 0
                            state = "get shape servo plan" 
                                   

        if state == "reset":   
            # rospy.loginfo("**Current state: " + state)
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            sample_count = 0
            execute_count = 0
            num_new_goal = 0
            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            print("Sucessfully reset robot and object")           
            state = "home"
 
  


        
        if plane_count >= max_plane_count:             
            all_done = True    

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)




    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)