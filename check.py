import numpy as np
import glob
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
object_class = os.path.basename(__file__)[:-4]

print(base_dir)

training = np.load('{}/training.npy'.format(base_dir))
testset = np.load('{}/testset.npy'.format(base_dir))

print(training.shape)
print(testset.shape)
