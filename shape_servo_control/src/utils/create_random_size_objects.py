import numpy as np
import trimesh
import os
import pickle

def create_tet(mesh_dir, object_name):
    # STL to mesh
    import os
    os.chdir('/home/baothach/fTetWild/build') 
    mesh_path = os.path.join(mesh_dir, object_name+'.stl')
    save_fTetwild_mesh_path = os.path.join(mesh_dir, object_name + '.mesh')
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path)


    # Mesh to tet:
    mesh_file = open(os.path.join(mesh_dir, object_name + '.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, object_name + '.tet'), "w")

    # Parse .mesh file
    mesh_lines = list(mesh_file)
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    vertices_start = mesh_lines.index('Vertices')
    num_vertices = mesh_lines[vertices_start + 1]

    vertices = mesh_lines[vertices_start + 2:vertices_start + 2
                        + int(num_vertices)]

    tetrahedra_start = mesh_lines.index('Tetrahedra')
    num_tetrahedra = mesh_lines[tetrahedra_start + 1]
    tetrahedra = mesh_lines[tetrahedra_start + 2:tetrahedra_start + 2
                            + int(num_tetrahedra)]

    print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

    # Write to tet output
    tet_output.write("# Tetrahedral mesh generated using\n\n")
    tet_output.write("# " + num_vertices + " vertices\n")
    for v in vertices:
        tet_output.write("v " + v + "\n")
    tet_output.write("\n")
    tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
    for t in tetrahedra:
        line = t.split(' 0')[0]
        line = line.split(" ")
        line = [str(int(k) - 1) for k in line]
        l_text = ' '.join(line)
        tet_output.write("t " + l_text + "\n")

mesh_dir = '/home/baothach/sim_data/Custom/Custom_mesh/multi_cylinders'
object_name = "cylinder_1"
# # #square_rod, round_rod, cylinder, hemis, thin_tisue_layer

# # # mesh2 = trimesh.creation.cylinder(radius=0.03, height=0.4)    # cylinder
mesh1 = trimesh.creation.cylinder(radius=0.08, height=0.2)
# # # mesh3 = trimesh.creation.cylinder(radius=0.06, height=0.23)
# # # mesh4 = trimesh.creation.cylinder(radius=0.03, height=0.23)
# # # mesh5 = trimesh.creation.cylinder(radius=0.03, height=0.5)
# # # mesh6 = trimesh.creation.cylinder(radius=0.1, height=0.6)
# # mesh7 = trimesh.creation.cylinder(radius=0.08, height=0.2)


# # mesh2 = trimesh.creation.box((0.1, 0.1, 0.04))      # box
# # # mesh1 = trimesh.creation.box((0.3, 0.15, 0.06))
# mesh3 = trimesh.creation.box((0.1, 0.1, 0.03))

# mesh1 = trimesh.creation.icosphere(radius = 0.3)   # hemisphere
# mesh1 = trimesh.intersections.slice_mesh_plane(mesh=mesh1, plane_normal=[0,0,1], plane_origin=[0,0,0.15], cap=True)
# mesh2 = trimesh.creation.icosphere(radius = 0.2)   # hemisphere
# mesh2 = trimesh.intersections.slice_mesh_plane(mesh=mesh2, plane_normal=[0,0,1], plane_origin=[0,0,0.1], cap=True)
# T = trimesh.transformations.translation_matrix([0., 0.6, 0])
# mesh2.apply_transform(T)

# meshes = [mesh1, mesh2]
# trimesh.Scene(meshes).show()
mesh1.export(os.path.join(mesh_dir, object_name+'.stl'))

create_tet(mesh_dir, object_name)





def create_cylinder_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True):
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        radius = np.random.uniform(low = 0.03, high = 0.06)
        height = np.random.uniform(low = 0.23, high = 0.40)
        mesh = trimesh.creation.cylinder(radius=radius, height=height)

        youngs_mean = 10000
        youngs_std = 1000        
        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "cylinder"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        
        primitive_dict[object_name] = {'radius': radius, 'height': height, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_cylinder.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def create_box_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True):
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        width = np.random.uniform(low = 0.1, high = 0.2)
        height = np.random.uniform(low = width, high = 0.3)
        thickness = np.random.uniform(low = 0.04, high = 0.06)
        mesh = trimesh.creation.box((height, width, thickness))

        youngs_mean = 1000
        youngs_std = 200        
        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "box"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def create_hemis_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True):
    primitive_dict = {'count':0}
    np.random.seed(0)
    for i in range(num_mesh):
        
        radius = np.random.uniform(low = 0.2, high = 0.3)
        origin = radius/0.2 * 0.1

        mesh = trimesh.creation.icosphere(radius = radius)   # hemisphere
        mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,origin], cap=True)

        youngs_mean = 1000
        youngs_std = 200        
        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "hemis"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet(save_mesh_dir, object_name)
        
        primitive_dict[object_name] = {'radius': radius, 'origin': origin, 'youngs': youngs}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_hemis.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


## 1000-200, 5000-1000, 10000-1000

# create_cylinder_mesh_datatset(mesh_dir)



# create_box_mesh_datatset(mesh_dir)


# create_hemis_mesh_datatset(mesh_dir)