import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

dataset = np.load('training_FPFH.npy')[::2000]
dataset1 = np.load(r'C:\Users\s143873\Desktop\Segmented point cloud\13\13_training.npy')[::300]

print(dataset[:,33:37])
print(dataset.shape)
print(dataset1.shape)


# m = MDS()
# xy = m.fit(dataset[:,0:37]).embedding_
#
# xs = xy[:,0]
# ys = xy[:,1]
#
# lblnms = [
#     "beam",
#     "ceiling",
#     "chair",
#     "column",
#     "door",
#     "floor",
#     "occlusion",
#     "table",
#     "wall",
#     "window"
# ]
#
# lbls = np.int64(dataset[:,-4])
#
# mp = plt.get_cmap("tab10")
#
# for i in range(np.amax(lbls) + 1):
#     mask = lbls == i
#     plt.scatter(xs[mask], ys[mask], c=mp(i), s=10, label=lblnms[i])
#
# plt.legend()
# plt.show()
# plt.clf()

m = TSNE()
xy = m.fit(dataset1[:,0:33]).embedding_

xs = xy[:,0]
ys = xy[:,1]

lblnms = [
    "beam",
    "ceiling",
    "chair",
    "column",
    "door",
    "floor",
    "occlusion",
    "table",
    "wall",
    "window"
]

lbls = np.int64(dataset1[:,-4])

mp = plt.get_cmap("tab10")

for i in range(np.amax(lbls) + 1):
    mask = lbls == i
    plt.scatter(xs[mask], ys[mask], c=mp(i), s=6, label=lblnms[i])

plt.legend()
plt.show()
plt.clf()
