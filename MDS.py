import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

dataset = np.load('training.npy')[::5000]
print(dataset)


m = MDS()
xy = m.fit(dataset[:,6:8]).embedding_

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

lbls = np.int64(dataset[:,-1])

mp = plt.get_cmap("tab10")

for i in range(np.amax(lbls) + 1):
    mask = lbls == i
    plt.scatter(xs[mask], ys[mask], c=mp(i), s=5, label=lblnms[i])

plt.legend()
plt.show()
plt.clf()
