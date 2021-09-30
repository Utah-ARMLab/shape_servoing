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

mesh_dir = '/home/baothach/sim_data/Custom/Custom_mesh/multi_boxes_test'
object_name = "box_4"
# # #square_rod, round_rod, cylinder, hemis, thin_tisue_layer

# # # mesh2 = trimesh.creation.cylinder(radius=0.03, height=0.4)    # cylinder
# # # mesh1 = trimesh.creation.cylinder(radius=0.08, height=0.2)
# # # mesh3 = trimesh.creation.cylinder(radius=0.06, height=0.23)
# # # mesh4 = trimesh.creation.cylinder(radius=0.03, height=0.23)
# # # mesh5 = trimesh.creation.cylinder(radius=0.03, height=0.5)
# # # mesh6 = trimesh.creation.cylinder(radius=0.1, height=0.6)
# # mesh7 = trimesh.creation.cylinder(radius=0.08, height=0.2)


# # mesh2 = trimesh.creation.box((0.1, 0.1, 0.04))      # box
# # # mesh1 = trimesh.creation.box((0.3, 0.15, 0.06))   # box
# mesh3 = trimesh.creation.box((0.1, 0.1, 0.03))
mesh4 = trimesh.creation.box((0.2, 0.2, 0.06))

# mesh1 = trimesh.creation.icosphere(radius = 0.3)   # hemisphere
# mesh1 = trimesh.intersections.slice_mesh_plane(mesh=mesh1, plane_normal=[0,0,1], plane_origin=[0,0,0.15], cap=True)
# mesh2 = trimesh.creation.icosphere(radius = 0.2)   # hemisphere
# mesh2 = trimesh.intersections.slice_mesh_plane(mesh=mesh2, plane_normal=[0,0,1], plane_origin=[0,0,0.1], cap=True)
# T = trimesh.transformations.translation_matrix([0., 0.6, 0])
# mesh2.apply_transform(T)

# meshes = [mesh1, mesh2]
# trimesh.Scene(meshes).show()
mesh4.export(os.path.join(mesh_dir, object_name+'.stl'))

create_tet(mesh_dir, object_name)





