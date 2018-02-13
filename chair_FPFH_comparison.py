import numpy as np
import pandas as pd

test = np.load('region_chair_test.npy')
train = np.load('region_chair_train.npy')

test_dataset = np.load('13_training.npy')
train_dataset = np.load(r'C:\Users\s143873\Desktop\Segmented point cloud\FPFH\4.npy')
np.asmatrix(train_dataset)

training = train_dataset[train]
testing = test_dataset[test]

asd = pd.DataFrame(training)
add = pd.DataFrame(testing)

asd.to_csv('chair_training_histogram.csv', sep=';')
add.to_csv('chair_testing_histogram.csv', sep=';')
