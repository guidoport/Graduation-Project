import numpy as np
import glob
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
# object_class = os.path.basename(__file__)[:-4]
# # print(glob.glob("*features.npy"))
file = glob.glob("*features.npy")
print(file)
# 
# # output = [np.concatenate(i) for i in [np.load('{}/{}'.format(base_dir, i)) for i in file]]
# # output1 = np.asarray(output)
# # print(output1)
# 
# 
# output = [np.load('{}/{}'.format(base_dir, i)) for i in file]
# for i in output:
# 	output1 = np.concatenate(i)
# 	
# print(output1)

v0 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[0]))
v1 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[1]))
v2 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[2]))
v3 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[3]))
v4 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[4]))
v5 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[5]))
v6 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[6]))
v7 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[7]))
v8 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[8]))
v9 = np.load('C:/Users/s143873/Desktop/Segmented point cloud/3/{}'.format(file[9]))

final = np.concatenate((v0, v1, v2, v3, v4, v5, v6, v7, v8, v9), axis=0)
print(final)
np.save('C:/Users/s143873/Desktop/Segmented point cloud/scene_3.npy', final)
