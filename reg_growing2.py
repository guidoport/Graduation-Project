import numpy as np
from pyflann import *
import itertools
import time
import pandas as pd

old_print = print

def print(*args):
    old_print(*args)
    time.sleep(0.001)

#class [0] #xyz [1:4] #pred_class [4] #pred_proba [5:15] #normals [15:18] #curv [18]
dataset = np.load('13_predicted_values_incl_probability.npy')
print(dataset.shape)
selection_index_list = [206213, 206693, 207554, 209749, 210762, 213457, 218289, 218981, 219426, 219918, 220582, 221523, 223156, 225818, 226242, 230390, 231265, 236947, 237118, 237240, 237438, 239301, 241981, 244333, 245133, 246079, 247959, 248523, 250027, 251688, 252396, 253000, 254351, 255141, 255416, 255450, 256795, 259204, 260454, 260732, 261227, 262051, 263020, 264377, 264924, 265084, 265554, 266346, 266431, 268321, 268837, 269093, 269989, 271248, 272417, 273077, 274661, 276721, 277708, 278539, 280185, 280821, 284002, 284147, 284453, 285389, 286236, 286517, 287259, 287458, 287760, 289797, 290200, 291441, 292352, 303026, 303151, 303206, 303594, 305052, 306301, 307077, 307915, 308031, 308544, 308942, 310317, 311869, 312667, 314774, 315126, 315444, 315681, 317162, 319022, 322729, 324878, 325183, 325595, 326063, 327041]
# print(selection_index_list.shape)

indices = selection_index_list
# df = pd.DataFrame(dataset[indices], columns=['class','x','y','z','true','prob1','prob2','prob3','prob4','prob5','prob6','prob7','prob8','prob9','prob10','nx','ny','nz','curv'])
df = pd.DataFrame(dataset[indices], columns=['class','x','y','z','true','prob1','prob2','prob3','prob4','prob5','prob6','prob7','prob8','prob9','prob10'])
# df.to_csv('points_not_in_region.csv', sep=';', float_format='%g')
#
# indices2 = [455970, 482526, 487349, 522050, 534147, 570850]
# df2 = pd.DataFrame(dataset[indices2], columns=['class','x','y','z','true','prob1','prob2','prob3','prob4','prob5','prob6','prob7','prob8','prob9','prob10','nx','ny','nz','curv'])
# df2.to_csv('points_in_region.csv', sep=';', float_format='%g')
print(df.mean(axis=0))
print(df.std(axis=0))
print(df['true'].value_counts())

quit()
selection_values = dataset[selection_index_list]
num_neighbors = 11
flann = FLANN()
flann.build_index(dataset[:,1:4], algorithm="kmeans", branching=32, iterations=7, checks=16)


def mean_std_calc(selection_values):
    mean = np.mean(selection_values, axis=0)
    std = np.std(selection_values, axis=0)
    return mean, std


def threshold(mean, std):
    multiplier = 2
    lower_threshold = mean - multiplier * std
    upper_threshold = mean + multiplier * std
    return lower_threshold, upper_threshold

# print(threshold(mean_std_calc(selection_values)[0], mean_std_calc(selection_values)[1]))


def find_neighbors(selection_index_list, num_neighbors):
    print("selection_index_list", selection_index_list, len(selection_index_list))
    print("dataset.shape", dataset.shape)
    result = flann.nn_index(dataset[selection_index_list, 1:4], num_neighbors=num_neighbors)
    print("result", result)
    # create_full_list = list(set(itertools.chain.from_iterable(result[0])))
    create_full_list = result[0].flatten()
    return create_full_list
    remove_dups = list(set(create_full_list).symmetric_difference(set(selection_index_list)))
    return remove_dups


# neighbor_list =

while True:
    ne_list = find_neighbors(selection_index_list, num_neighbors)


variable = 13
value_for_filter = (dataset[neighbor_list[0]])
print(value_for_filter)
value_lower_threshold = (threshold(mean_std_calc(selection_values)[0], mean_std_calc(selection_values)[1])[0])
print(value_lower_threshold)
value_upper_threshold = (threshold(mean_std_calc(selection_values)[0], mean_std_calc(selection_values)[1])[1])
print(value_upper_threshold)

# b = a[a>threshold]

def filter_threshold():
    s = set()
    for i in neighbor_list:
        if dataset[i, variable] >= value_lower_threshold[variable] and dataset[i, variable] <= value_upper_threshold[variable]:
            s.add(i)
    if len(s) == 0:
        empty_set = True
    else:
        empty_set = False
    return s, empty_set


filter_threshold()


def region_grow(selection_index_list):
    while True:
        ne_et, empty = filter_threshold()
        if empty: break
        s = selection_index_list
        # for i in s:
        find_neighbors(s, num_neighbors)
        selection_index_list.extend(ne_et)
    return selection_index_list


region_grow(selection_index_list)

quit()

# def filter_threshold():
#     s = set()
#     variable = 13  # wall proba
#     for i in neighbor_list:
#         if dataset[i,variable] >= threshold_neg[variable] and dataset[i,variable] <= threshold_pos:
#             s.add(i)
#     return s
#
# print(filter_threshold())
