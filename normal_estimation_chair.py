from pyflann import *
from scipy.odr import ODR, Model, Data, Output
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

# Import the dataset and testset (preselection)
ds = pd.read_csv('C:/Users/s143873/Desktop/Segmented point cloud/11+12/chair.csv', delimiter=';')
ds.columns = ['x', 'y', 'z']
ts = pd.read_csv('C:/Users/s143873/Desktop/Segmented point cloud/11+12/chair.csv', delimiter=';')

print(ds)

# Transform dataframes to matrices
dataset = ds.as_matrix()
testset = ts.as_matrix()

size = len(dataset)

flann = FLANN()
flann.build_index(dataset)

# Function for finding n-nearest neighbours
def NN(t, nn):
    if t % 1000 == 0:
        print(t * 100 / size)

    point_in_dataset = testset[t]

    # flann = FLANN()
    # result, dists = flann.nn(dataset, point_in_dataset, nn, algorithm="kmeans", branching=2, iterations=7, checks=16)
    result, dists = flann.nn_index(point_in_dataset, nn)
    a = result[0]
    pc_0 = pd.DataFrame(dataset[a])
    pc_0.columns = ['x', 'y', 'z']
    return pc_0

# Function for normal vector calculation -------------------------------------------------------------------------------
def Normal_calc(t, nn):
    pc_0 = NN(t, nn)
    x = pc_0.x
    y = pc_0.y
    z = pc_0.z

    def func(beta, data):
        x, y = data
        a, b, c = beta
        return a * x + b * y + c

    data = Data([x, y], z)
    model = Model(func)
    odr = ODR(data, model, beta0=[0.0, 0.0, 0.0])
    odr.set_job(fit_type=0)
    res = odr.run()

    """Extend plot with plt.Quiver (vectors) later on...?"""

    # Calculate xyz coordinates for corner vertices of the plane
    Y, X = np.mgrid[y.min():y.max():2j, x.min():x.max():2j]
    Z = func(res.beta, [X, Y])
    # f = plt.figure()
    # pl = f.add_subplot(111, projection='3d')
    # pl.scatter3D(x, y, z)
    # pl.plot_surface(X, Y, Z, alpha=0.4)

    # Define 3 points on plane for cross product calculation (from previous calculation)
    P = [X[0][0], Y[0][0], Z[0][0]]
    Q = [X[0][1], Y[0][1], Z[0][1]]
    R = [X[1][0], Y[1][0], Z[1][0]]

    # Calculate vectors on plane
    PQ = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]]
    PR = [R[0] - P[0], R[1] - P[1], R[2] - P[2]]

    # Calculate cross product of vectors + normalize to 1 = sqrt(x**2+y**2+z**2)
    N1 = np.cross(PQ, PR)
    N1_array = np.array([[N1[0], N1[1], N1[2]]], dtype=np.float)
    N1_normalized = preprocessing.normalize(N1_array, norm='l2')
    return N1_normalized[0]

result_norm = np.array([Normal_calc(i, 9) for i in range(len(dataset))])
norm_out = pd.DataFrame(result_norm, columns = ['n_x', 'n_y', 'n_z'])

# Curvature estimation -------------------------------------------------------------------------------------------------

def NN_curv(t, nn):
    point_in_dataset = dataset[t]
    result, dists = flann.nn_index(point_in_dataset, nn + 1)
    return result[:,1:]

num_neighbours = 9
A = np.repeat(dataset, 9, axis=0)
A.shape = (dataset.shape[0], num_neighbours, 3)
n = NN_curv(list(range(dataset.shape[0])),num_neighbours)
B = dataset[n]

p = A - B
# print(p.shape)

C = np.zeros((dataset.shape[0], 6))
C[:,0] = np.sum(p[:,:,1-1] * p[:,:,1-1],axis=1)
C[:,1] = np.sum(p[:,:,1-1] * p[:,:,2-1],axis=1)
C[:,2] = np.sum(p[:,:,1-1] * p[:,:,3-1],axis=1)
C[:,3] = np.sum(p[:,:,2-1] * p[:,:,2-1],axis=1)
C[:,4] = np.sum(p[:,:,2-1] * p[:,:,3-1],axis=1)
C[:,5] = np.sum(p[:,:,3-1] * p[:,:,3-1],axis=1)
C /= num_neighbours

# print(C)

def curv_estim(i):
    Cmat = [[C[i, 0], C[i, 1], C[i, 2]],
            [C[i, 1], C[i, 3], C[i, 4]],
            [C[i, 2], C[i, 4], C[i, 5]]]
    D, V = linalg.eig(Cmat)
    D = np.flipud(D)
    lambda0 = np.min(D)
    curvature = lambda0 / np.sum(D)
    return curvature

curvs = np.array([curv_estim(i) for i in range(len(dataset))])
curvs_out = pd.DataFrame(curvs, columns = ['c'])

# Horizontal/vertical/diagonal residue ---------------------------------------------------------------------------------
orientation_l = []
for i in result_norm:
    if i[2] >= 0.97 or i[2] <= -0.97:
        orientation_l.append('0') # horizontal
    elif i[2] <= 0.03 and i[2] >= -0.03:
        orientation_l.append('1') # vertical
    else:
        orientation_l.append('2') # diagonal

orientation_l = np.array(orientation_l)
orient_out = pd.DataFrame(orientation_l, columns = ['orientation'])

# Concatenation final output -------------------------------------------------------------------------------------------
print(ds)
print(norm_out)
print(curvs_out)

final_output = pd.concat([ds, norm_out, curvs_out, orient_out], axis=1)
final_output.to_csv('C:/Users/s143873/Desktop/Segmented point cloud/11+12/include_normals/chair.csv')


quit()
# ------------------------------------------------------
x = [item[0] for item in dataset]
y = [item[1] for item in dataset]
z = [item[2] for item in dataset]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()
