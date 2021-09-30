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


    # Some important paramters
    init_dvrk_joints(gym, envs[i], dvrk_handles[i])  # Initilize robot's joints    

    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 

    all_done = False
    state = "home"
    goal_datas = []
    goal_point_count = 0
    max_goal_count = 10 #10
    
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0

    max_group_count = 10
    max_sample_count = 1

    dc_client = GraspClient()
    fail_mtp = False


    
    start_time = timeit.default_timer()   
    close_viewer = False
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
                
                # Get current object state
                current_particle_state = get_particle_state()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_particle_state))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)

                # Go to next state
                state = "generate preshape"


        ############################################################################
        # generate preshape: Sample a new manipulation point
        ############################################################################
        if state == "generate preshape":                   

            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET-0.02,
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
        
                current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                
                # Save robot and object states for reset
                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                shapesrv_start_time = timeit.default_timer()

                state = "get shape servo plan"



        ############################################################################
        # get shape servo plan: sample random delta x, y, z and set up MoveIt
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            # Set limits so robot doesn't move too much
            max_x = max_y = max_z = min(h, 0.3) * 0.7 * 0.8   

            # Sample random delta x, y, z
            delta_x = np.random.uniform(low = -max_x, high = max_x)
            delta_y = np.random.uniform(low = 0.0, high = max_y)
            delta_z = np.random.uniform(low = 0.00, high = max_z)     

            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
            cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z

            # Set up moveit for the above delta x, y, z
            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to shape servo. Ignore this grasp.\n')  
                state = "reset"
            else:
                state = "move to goal"
                traj_index = 0


        ############################################################################
        # move to goal: Move robot gripper to the desired delta x, y, z using MoveIt
        ############################################################################
        if state == "move to goal":
            # lose contact w robot
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  
                print("Lost contact with robot")                
                # group_count += 1
                state = "reset"

            else:     
                dof_states = robot.get_full_joint_positions()
                plan_traj_with_gripper = [plan+[g_1_pos,g_2_pos] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                
                

                if traj_index <= len(plan_traj) - 2:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                        traj_index += 1 
                else:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                        traj_index += 1   

                if traj_index == len(plan_traj):
                    traj_index = 0  
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  
                    
                    pc_goal = get_partial_point_cloud(gym, sim, envs[0], cam_handle, cam_prop)
                    particle_state = get_particle_state()
                    final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)[-3])                    

                    goal_data = {"particle state": particle_state, "partial pc": pc_goal, "positions": [delta_x, delta_y, delta_z]}
                    goal_datas.append(goal_data)
                    goal_point_count += 1
                    
                    # Reset everything
                    frame_count = 0
                    # terminate_count = 0
                    # sample_count += 1
                    # print("group ", group_count, ", sample ", sample_count)
   
                    state = "get shape servo plan"

        ############################################################################
        # grasp object: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            sample_count = 0
            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            print("Sucessfully reset robot and object")                
            
            state = "get shape servo plan"
            # Failed move to preshape -> back to home
            if fail_mtp:
                state = "home"  
                fail_mtp = False
        

        if  goal_point_count >= max_goal_count:                     
            all_done = True 
            final_data = goal_datas
            with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
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
    print("total data pt count: ", goal_point_count)
