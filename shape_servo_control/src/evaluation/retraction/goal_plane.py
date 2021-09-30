import open3d
import pyransac3d as pyrsc
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def vecalign(a, b):
    '''
    Returns the rotation matrix that can rotate the 3 dimensional b vector to
    be aligned with the a vector.
    @param a (3-dim array-like): destination vector
    @param b (3-dim array-like): vector to rotate align with a in direction
    the vectors a and b do not need to be normalized.  They can be column
    vectors or row vectors
    '''
    with np.errstate(divide='raise', under='raise', over='raise', invalid='raise'):
        a = np.asarray(a)
        b = np.asarray(b)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cos_theta = a.dot(b) # since a and be are unit vecs, a.dot(b) = cos(theta)

        # if dot is close to -1, vectors are nearly equal
        if np.isclose(cos_theta, 1):
            # TODO: solve better than just assuming they are exactly identical
            return np.eye(3)

        # if dot is close to -1, vectors are nearly opposites
        if np.isclose(cos_theta, -1):
            # TODO: solve better than just assuming they are exactly opposite
            return -np.eye(3)

        axis = np.cross(b, a)
        sin_theta = np.linalg.norm(axis)
        axis = axis / sin_theta
        c = cos_theta
        s = sin_theta
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]

        # angle-axis formula to create a rotation matrix
        return np.array([
            [t*x*x + c  , t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c  , t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ],
            ])


def get_goal_plane(constrain_plane, initial_pc, check = False, delta = 0.01, current_pc=[]):

    # Check if retraction is already successful or not 
    if check:
        failed_points = np.array([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])                   
        if len(failed_points) == 0:
            return 'success'
        else:
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(failed_points)
            pcd2.paint_uniform_color([1,0,0])           
            constrain_plane = constrain_plane.copy()
            constrain_plane[3] += delta

    # Offset the target plane a small amount for faster retraction
    constrain_plane = constrain_plane.copy()
    constrain_plane[3] += 0.03

    # Get failed/passed points 
    failed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])
    passed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] <= -constrain_plane[3]])

    if len(passed_points) == 0:
        return None

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(failed_points)
    pcd2.paint_uniform_color([1,0,0])
    # open3d.visualization.draw_geometries([pcd2])

 
    # Find rotation point:
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
    center = pcd.get_center().reshape(3)
    dist = (np.dot(center, constrain_plane[:3]) + constrain_plane[3])/np.linalg.norm(constrain_plane[:3])
    unit_normal = np.array(constrain_plane[:3])/np.linalg.norm(constrain_plane[:3])
    rot_pt = center - dist*unit_normal

    pcd.points = open3d.utility.Vector3dVector(passed_points)
    pcd.paint_uniform_color([0,0,0])


    # Fit a plane to the failed_points
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(failed_points, thresh=0.001, maxIteration=1000)
    if best_eq[2] > 0:
        best_eq = -np.array(best_eq)

    # Find rotation between the above plane and the target plane 
    r = vecalign(np.array(best_eq)[:3], constrain_plane[:3])
    
    # Rotate failed points
    pcd2.rotate(R = r.T, center = rot_pt.reshape((3,1)))

    # Get heuristic goal pc
    goal_pcd = pcd + pcd2
    # open3d.visualization.draw_geometries([goal_pcd])
    
    return np.asarray(goal_pcd.points)



# def generate_new_target_plane():
#     choice = np.random.randint(0,3)
#     if choice == 0: # horizontal planes
#         pos = np.random.uniform(low=0.45, high=0.50)
#         return np.array([0, 1, 0, pos])
#     elif choice == 1:   # tilted left
#         pos = np.random.uniform(low=0.38, high=0.45) 
#         return np.array([1, 1, 0, pos])
#     elif choice == 2:   # tilted right
#         pos = np.random.uniform(low=0.45, high=0.50)  
#         return np.array([-1, 1, 0, pos]) 