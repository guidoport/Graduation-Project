from pyflann import *
from scipy.odr import ODR, Model, Data, Output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn import preprocessing


base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

# def create_grid():
#     x = 0
#     y = np.arange(20)
#     z = np.arange(20)
#     X,Y,Z = np.meshgrid(x,y,z)
#     XYZ = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
#     a = np.arange(20)
#     b = np.arange(20)
#     c = 0
#     A,B,C = np.meshgrid(a,b,c)
#     ABC = np.array([A.flatten(),B.flatten(),C.flatten()]).T
#     u = np.arange(20)
#     v = np.arange(20)
#     w = 19
#     U,V,W = np.meshgrid(u,v,w)
#     UVW = np.array([U.flatten(),V.flatten(),W.flatten()]).T
#     dataset = np.concatenate((XYZ, ABC, UVW), axis=0)
#     unique_rows = np.unique(dataset, axis=0)
#     return unique_rows
#
# dataset = create_grid()
# np.save('{}/hist_test_data.npy'.format(base_dir), dataset)

dataset = np.load('{}/hist_test_data.npy'.format(base_dir))
dataset = dataset+1
testset = dataset

flann = FLANN()
flann.build_index(dataset)

# Function for finding n-nearest neighbours
def NN(t, nn):
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
    f = plt.figure()
    pl = f.add_subplot(111, projection='3d')
    pl.scatter3D(x, y, z)
    pl.plot_surface(X, Y, Z, alpha=0.4)
    plt.show()

    # Define 3 points on plane for cross product calculation (from previous calculation)
    P = [X[0][0], Y[0][0], Z[0][0]]
    Q = [X[0][1], Y[0][1], Z[0][1]]
    R = [X[1][0], Y[1][0], Z[1][0]]
    print('PQR:', P, Q, R)

    # Calculate vectors on plane
    PQ = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]]
    PR = [R[0] - P[0], R[1] - P[1], R[2] - P[2]]
    print(PQ, PR)

    # Calculate cross product of vectors + normalize to 1 = sqrt(x**2+y**2+z**2)
    N1 = np.cross(PQ, PR)
    print('N1:', N1)
    N1_array = np.array([[N1[0], N1[1], N1[2]]], dtype=np.float)
    N1_normalized = preprocessing.normalize(N1_array, norm='l2')
    return N1_normalized[0]

# result_norm = np.array([Normal_calc(i, 9) for i in range(len(dataset))], dtype=object)

test_point = Normal_calc(1000, 11)
print(test_point)
quit()
print(result_norm[0:50])

print(dataset[0:50])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = dataset[:,0]
y = dataset[:,1]
z = dataset[:,2]
u = result_norm[:,0]
v = result_norm[:,1]
w = result_norm[:,2]
ax.scatter(x,y,z)
ax.quiver(x, y, z, u, v, w, length = 0.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

