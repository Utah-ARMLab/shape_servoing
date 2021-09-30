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

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl

sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
from architecture import DeformerNet


import torch

ROBOT_Z_OFFSET = 0.25




if __name__ == "__main__":

    main_path = "/home/baothach/shape_servo_data/evaluation"

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="dvrk",
        custom_parameters=[
            {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
            {"name": "--obj_type", "type": str, "default": 'box_1k', "help": "box1k, box5k, box10k"},
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    args.headless = args.headless == "True"
    args.inside = args.inside == "True"


    # configure sim
    sim, sim_params = default_sim_config(gym, args)


    # Get primitive shape dictionary to know the dimension of the object   
    if args.inside:
        object_meshes_path = os.path.join(main_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(main_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    h = data["height"]
    w = data["width"]
    thickness = data["thickness"]


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
    if args.inside:
        asset_root = os.path.join(main_path, "urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(main_path, "urdf", args.obj_type, "outside") 

    soft_asset_file = args.obj_name + ".urdf"    
    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.7)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.0005#0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)       


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



    # DOF Properties and Drive Modes 
    dof_props = gym.get_actor_dof_properties(envs[0], dvrk_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)  
    vel_limits = dof_props['velocity'] 

    
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
    
    if args.obj_type == "box_1k":
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/weights/run1"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 362")))  
    elif args.obj_type == "box_5k":
        # weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run1"
        # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 250")))  
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run3(partial)"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))              
    elif args.obj_type == "box_10k":
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_10kPa/weights/run1"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 288"))) 

    model.eval()


    # Some important paramters
    all_done = False
    state = "home"
    first_time = True  

    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
        chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 
        chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "outside")

    goal_count = 0 #0
    frame_count = 0
    max_goal_count =  2#10

    max_shapesrv_time = 2*60    # 2 mins
    if args.inside:
        min_chamfer_dist = 0.2
    else:
        min_chamfer_dist = 0.2 #0.25
    fail_mtp = False
    saved_chamfers = []
    final_chamfer_distances = []      

    dc_client = GraspClient()   
    start_time = timeit.default_timer()    
    close_viewer = False

   
    # Get 10 goal pc data for 1 object:
    with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = goal_datas[goal_count]["partial pc"]   # first goal pc
    goal_pc_numpy = down_sampling(goal_pc_numpy)
    goal_pc = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float() 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
    goal_position = goal_datas[goal_count]["positions"]    


    # Set up robot
    print("vel_limits:", vel_limits) 
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.203)            
            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state)
                frame_count = 0

                # Save robot and object states for reset 
                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))           
                
                # Go to next state
                state = "generate preshape"


        ############################################################################
        # generate preshape: Obtain the computed anipulation point
        ############################################################################
        if state == "generate preshape":                   

            rospy.loginfo("**Current state: " + state)      
            mp_location = goal_datas[goal_count]["mani point"]
            target_pose = [-mp_location[0], -mp_location[1], mp_location[2]-ROBOT_Z_OFFSET-0.02,
                            0, 0.707107, 0.707107, 0]


            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"  
                fail_mtp = True              
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


        ############################################################################
        # move to preshape: Move robot gripper to the manipulation point location
        ############################################################################
        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's grasp it!!") 


        ############################################################################
        # grasp object: close gripper
        ############################################################################        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:                                     
                                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), g_2_pos)         
        
                init_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                
                # Save robot and object states for reset
                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                shapesrv_start_time = timeit.default_timer()

                # Switch to velocity control mode
                dof_props['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props["stiffness"][:8].fill(0.0)
                dof_props["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, dvrk_handles[i], dof_props)


                state = "get shape servo plan"



        ############################################################################
        # get shape servo plan: compute delta x, y, z with DeformerNet
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            # Get current gripper pose
            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
            print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 

            # Get current point cloud
            current_pc = down_sampling(get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop))                

            # Calculate Chamfer dist                
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc)  
            ## open3d.visualization.draw_geometries([pcd, pcd_goal])  
            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            print("chamfer distance: ", chamfer_dist)

            if chamfer_dist >= min_chamfer_dist:
                current_pc = torch.from_numpy(current_pc).permute(1,0).unsqueeze(0).float()
                with torch.no_grad():
                    desired_position = model(current_pc, goal_pc)[0].detach().numpy()*(0.001) 
            print("from model:", desired_position)
            print("ground truth: ", goal_position)             


            # Set up resolved rate controller
            tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits)     

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))


            state = "move to goal"


        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using resolved rate controlller
        ############################################################################
        if state == "move to goal":
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                print("Lost contact with robot")
                state = "reset" 
                final_chamfer_distances.append(999) 
                goal_count += 1
            
            if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                print("Timeout")
                state = "reset" 
                current_pc = get_particle_state()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(current_pc)   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(min(saved_chamfers)) 
                goal_count += 1

            else:
                action = tvc_behavior.get_action()  
                # print("==test timer:", timeit.default_timer()-test)
                if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 1.5:   
                    final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
                    delta_x = -(final_pose["pose"]["p"]["x"] - init_pose["pose"]["p"]["x"])
                    delta_y = -(final_pose["pose"]["p"]["y"] - init_pose["pose"]["p"]["y"])
                    delta_z = final_pose["pose"]["p"]["z"] - init_pose["pose"]["p"]["z"]
                    print("delta x, y, z:", delta_x, delta_y, delta_z)
                    state = "get shape servo plan"    
                else:
                    # print("desired vel:", action.get_joint_position())
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())        

                # Terminal conditions
                if all(abs(desired_position) <= 0.005) \
                        or chamfer_dist < min_chamfer_dist:
                    
                    current_pc = down_sampling(get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop))  
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)  
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                    print("final chamfer distance: ", chamfer_dist)
                    final_chamfer_distances.append(chamfer_dist) 
                    goal_count += 1

                    state = "reset" 

        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            saved_chamfers = []
            
            rospy.loginfo(("=== JUST ENDED goal_count" + str(goal_count)))

            # Reset
            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            print("Sucessfully reset robot and object")

            # Restart time
            shapesrv_start_time = timeit.default_timer()            
            
            # Go to next goal pc
            if goal_count < max_goal_count:
                goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pc"])
                goal_pc = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float() 
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                goal_position = goal_datas[goal_count]["positions"]               
           
            if fail_mtp:
                state = "home"  
                fail_mtp = False

            state = "get shape servo plan"

        ############################################################################
        # Record all final Chamfer distances
        ############################################################################  
        if  goal_count >= max_goal_count:                    
            all_done = True 
            final_data = final_chamfer_distances
            with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

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
    # print("total data pt count: ", data_point_count)
