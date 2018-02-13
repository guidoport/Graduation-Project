import numpy as np
import pandas as pd

pred = np.load ('13_predicted_values.npy')

print(pred)

actual = pred[:,0].astype(int)
xyz = pred[:,1:4]
predict = pred[:,4].astype(int)
compare = np.equal(actual,predict)*1

print(compare)

final = np.array(xyz, dtype=object)
final = np.append(final, actual[:,None], axis=1)
final = np.append(final, predict[:,None], axis=1)
final = np.append(final, compare[:,None], axis=1)
# final = np.concatenate((pred, compare[:,None]), axis=1,)
print(final)


def create_ply(data, name):
    outputFile = '{}.ply'.format(name)
    lines = [" ".join([ str(v) for v in i ]) + "\n" for i in data]
    f = open(outputFile, 'w')
    f.writelines("ply\n")
    f.writelines("format ascii 1.0\n")
    f.writelines("comment Created by Guido: use for blender\n")
    f.writelines("element vertex {}\n".format(len(lines)))
    f.writelines("property float x\n"
                 "property float y\n"
                 "property float z\n"
                 "property int class\n"
                 "property int pred\n"
                 "property int diff\n")
    f.writelines("end_header\n")
    f.writelines(lines)
    f.close()


create_ply(final, '13_predicted')