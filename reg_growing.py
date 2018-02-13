import numpy as np
from pyflann import *
import time

old_print = print

def print(*args):
    old_print(*args)
    time.sleep(0.001)

class Queue:
    def __init__(self):
        self.items = []

    def __str__(self):
        return str(self.items)

    def __getitem__(self, k):
        return self.items[k]

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def isInside(self, item):
        return (item in self.items)



def grow_region(seed, amount_NNs):
    q = Queue()
    s = set()
    q.enqueue(seed)
    print("the seed point is:", q)
    print("the amount of NNs - 1 is:", amount_NNs - 1)

    while not q.isEmpty():
        print("---- currently in queue", q)
        x = q.dequeue()
        print("---- current NN search for point index:", x)

        if x not in s:
            s.add(x)
            nns = NN_index(amount_NNs, x)
            print()

            for n in nns:
                if n not in s:

                    q.enqueue(n)
    return s

#class [0] #xyz [1:4] #pred_class [4] #pred_proba [5:14]
dataset = np.load('19+20_predicted_values_incl_probability.npy')
selection_index_list = np.load('selection_region_growing.npy')
print(dataset[0])
print(dataset[:,5:15])

def calc_mean_std(selection_index_list, dataset):
    selection_values = dataset[selection_index_list]
    mean = np.mean(selection_values, axis=0)
    std = np.std(selection_values, axis=0)
    return mean, std

print(calc_mean_std(selection_index_list, dataset))
# selection_values = dataset[selection]
# print(selection_values.shape)
# selection_values = np.array(selection_values)
# mean = np.mean(selection_values, axis=0)
# std = np.std(selection_values, axis=0)
# print(mean)
# print(std)

# unique, counts = np.unique(selection_values[:,4], return_counts=True)
# asd = dict(zip(unique, counts))
# print(asd)

quit()
flann = FLANN()
flann.build_index(dataset[:,1:4], algorithm="kmeans", branching=32, iterations=7, checks=16)
result = flann.nn_index(dataset[0,1:4], num_neighbors = 9)
print(result)