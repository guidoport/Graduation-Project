import numpy as np

import os
import pandas as pd
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

import os

class_dict = {"beam":0, "ceil":1, "chai":2, "colu":3, "door":4, "floo":5, "occl":6, "tabl":7, "wall":8, "wind":9}
print(class_dict["beam"])

scene_num = '19+20'

outputFile = 'C:/Users/s143873/Desktop/Segmented point cloud/FPFH/{}.ply'.format(scene_num)

# beam = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/beam.npy'.format(scene_num))
ceil = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/ceiling.npy'.format(scene_num))
chai = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/chair.npy'.format(scene_num))
# colu = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/column.npy'.format(scene_num))
door = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/door.npy'.format(scene_num))
floo = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/floor.npy'.format(scene_num))
occl = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/occlusion.npy'.format(scene_num))
tabl = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/table.npy'.format(scene_num))
wall = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/wall.npy'.format(scene_num))
wind = np.load('C:/Users/s143873/Desktop/Segmented point cloud/{}/window.npy'.format(scene_num))

elem_length = len(ceil)+len(chai)+len(door)+len(floo)+len(occl)+len(tabl)+len(wall)+len(wind)
print(elem_length)


# lines_beam = ["{}".format(class_dict["beam"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in beam]
lines_ceil = ["{}".format(class_dict["ceil"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in ceil]
lines_chai = ["{}".format(class_dict["chai"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in chai]
# lines_colu = ["{}".format(class_dict["colu"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in colu]
lines_door = ["{}".format(class_dict["door"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in door]
lines_floo = ["{}".format(class_dict["floo"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in floo]
lines_occl = ["{}".format(class_dict["occl"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in occl]
lines_tabl = ["{}".format(class_dict["tabl"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in tabl]
lines_wall = ["{}".format(class_dict["wall"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in wall]
lines_wind = ["{}".format(class_dict["wind"]) +" " + " ".join([ str(v) for v in i ]) + "\n" for i in wind]

# csvLines = [" ".join([ str(v) for v in i ]) + "\n" for i in dataset]
print(lines_ceil)

f = open( outputFile, 'w' )
f.writelines("ply\n")
f.writelines("format ascii 1.0\n")
f.writelines("comment Created by Guido: used for FPFH desciptor\n")
f.writelines("element vertex {}\n".format(elem_length))
f.writelines("property int class\n"
             "property float x\n"
             "property float y\n"
             "property float z\n")
f.writelines("end_header\n")
# f.writelines(lines_beam)
f.writelines(lines_ceil)
f.writelines(lines_chai)
# f.writelines(lines_colu)
f.writelines(lines_door)
f.writelines(lines_floo)
f.writelines(lines_occl)
f.writelines(lines_tabl)
f.writelines(lines_wall)
f.writelines(lines_wind)
f.close()

quit()

# transform ply to pcd
# FPFH 33 - normal xyz 3 - curv 1 - class 1 - xyz - 3

# pc = np.genfromtxt('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/11+12.pcd', dtype=None)
# pc1 = np.save('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/11+12.npy', pc)

# pc0 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/3.npy')
# pc1 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/4.npy')
# pc2 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/6+7.npy')
# pc3 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/9+10.npy')
# pc4 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/11+12.npy')
#
#
#
# final = np.concatenate((pc0, pc1, pc2, pc3, pc4,), axis=0)
# print(final)
# print(final.shape)
#
# np.save('{}/training_FPFH.npy'.format(base_dir), final)


# print(pc[0:10])

# pc = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/training_FPFH.npy')
# pc1 = pc[0]
# print(isinstance(pc1, object))
# print(pc1[0])
# df = pd.DataFrame(pc)
#
# asd = df.as_matrix()
# print(asd)
#
# np.save('C:/Users/s143873/Desktop/Segmented point cloud/FPFH/training_FPFH.npy', asd)
