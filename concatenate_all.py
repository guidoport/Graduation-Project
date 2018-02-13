import numpy as np
import glob
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
object_class = os.path.basename(__file__)[:-4]
# print(glob.glob("*features.npy"))
file = ["beam_features.npy",
        "ceiling_features.npy",
        "chair_features.npy",
        "column_features.npy",
        "door_features.npy",
        "floor_features.npy",
        "occlusion_features.npy",
        "table_features.npy",
        "wall_features.npy",
        "window_features.npy"]


def full_out(scene_number):
    output = [np.load('{}/{}/{}'.format(base_dir,scene_number, i)) for i in file]
    output1 = np.concatenate(output, axis=0)
    return output1


final = np.concatenate((full_out('3'), full_out('4'), full_out('6+7'), full_out('9+10'), full_out('11+12')), axis=0)
print(final.shape)

np.save('{}/training.npy'.format(base_dir), final)