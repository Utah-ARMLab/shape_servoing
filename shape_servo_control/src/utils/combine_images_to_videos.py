#!/usr/bin/env python3
import os

# img_dir = "/home/baothach/shape_servo_data/new_task/plane_vis/6_4"
img_dir = "/home/baothach/shape_servo_data/visualization/cylinder/1_projected"
os.chdir(img_dir)


os.system("ffmpeg -framerate 15 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out_sim_goal_oriented_cylinder.mp4")
# os.system("ffmpeg -framerate 20 -i img%04d.png -pix_fmt yuv420p output_sim_goal_oriented.mp4")
