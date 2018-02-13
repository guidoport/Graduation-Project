import numpy as np

import os
import pandas as pd
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

# transform ply to pcd
# FPFH 33 - normal xyz 3 - curv 1 - class 1 - xyz - 3

# pc = np.genfromtxt('C:/Users/s143873/Desktop/Segmented point cloud/FPFH - global/11+12.pcd', dtype=None)
# pc1 = np.save('C:/Users/s143873/Desktop/Segmented point cloud/FPFH - global/11+12.npy', pc)

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
# np.save('{}/training_FPFH_global.npy'.format(base_dir), final)


# print(pc[0:10])
#
pc = np.load('C:/Users/s143873/Desktop/Segmented point cloud/FPFH - global/training_FPFH_global.npy')
pc1 = pc[0]
print(isinstance(pc1, object))
print(pc1[0])
df = pd.DataFrame(pc)

asd = df.as_matrix()
print(asd)

np.save('C:/Users/s143873/Desktop/Segmented point cloud/FPFH - global/training_FPFH_global.npy', asd)
