import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = np.load('training_FPFH.npy')
print(dataset[0,33:42])

def remove_nan(a):
    return a[~np.isnan(a).any(axis=1)]


dataset = remove_nan(dataset)

dataset = np.delete(dataset, (2887663, 3267169, 2865900, 2764570, 2846506, 2899855, 2879309), axis=0)



dataset_z = dataset[:,40]
print(dataset_z)

z_max = np.amax(dataset_z)
z_min = np.amin(dataset_z)
print(z_min, z_max)

def height_feature(i, z_max, z_min):
    z = i
    h_below = z - z_min
    h_above = z_max - z
    return h_below, h_above

out = np.array(height_feature(dataset_z, z_max, z_min))
print(out.shape)
print(dataset.shape)

print(out)

out_tr = out.transpose()
print(out_tr)


new_dataset = np.concatenate((dataset, out_tr), axis=1)
print(new_dataset)

np.save('training_FPFH_height.npy', new_dataset)