import numpy as np
import glob
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
object_class = os.path.basename(__file__)[:-4]
# print(glob.glob("*features.npy"))
file = ["beam_features.npy",
        "ceiling_features.npy",
        "chair_features.npy",
        "floor_features.npy",
        "occlusion_features.npy",
        "table_features.npy",
        "wall_features.npy",
        "window_features.npy"]


def full_out(scene_number):
    output = [np.load('{}/{}/{}'.format(base_dir,scene_number, i)) for i in file]
    output1 = np.concatenate(output, axis=0)
    return output1


final = full_out('13')
print(final.shape)

np.save('{}/testset.npy'.format(base_dir), final)