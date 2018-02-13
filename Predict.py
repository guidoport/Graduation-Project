import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras import metrics
from keras import regularizers
import pandas as pd
import h5py


model = load_model('neural_net_FPFH.h5')

dataset = np.load('19+20_training.npy')

fpfh_norm = np.divide(dataset[:,0:33], 100)
normals_curv = dataset[:,33:37]
spatial_xyz = dataset[:,38:41]

X_test = np.concatenate((fpfh_norm, normals_curv), axis=1)
print(X_test.shape)

predictions = model.predict_classes(X_test, batch_size=10000)
probabilities = model.predict_proba(X_test, batch_size=10000)
Y_test = (dataset[:,37])

# print(predictions) # predicted
# print(Y_test)      # actual

print(probabilities)
print(probabilities.shape)

print(sum(1 for x,y in zip(Y_test,predictions) if x == y) / len(Y_test))


classified_data = np.concatenate((dataset[:,37:41], predictions[:,None], probabilities, dataset[:,33:37]), axis=1)
print(classified_data)
np.save('19+20_predicted_values_incl_probability.npy', classified_data)
quit()

# class_dict = {"beam":0, "ceiling":1, "chair":2, "column":3, "door":4, "floor":5, "occlusion":6, "table":7, "wall":8, "window":9}
#
# df = pd.DataFrame(classified_data, columns=['x', 'y', 'z', 'class'])
# class_beam = df[df['class'] == 0]
# class_ceiling = df[df['class'] == 1]
# class_chair = df[df['class'] == 2]
# class_column = df[df['class'] == 3]
# class_door = df[df['class'] == 4]
# class_floor = df[df['class'] == 5]
# class_occlusion = df[df['class'] == 6]
# class_table = df[df['class'] == 7]
# class_wall = df[df['class'] == 8]
# class_window = df[df['class'] == 9]
#
# beam = class_beam.drop('class', 1).as_matrix()
# ceiling = class_ceiling.drop('class', 1).as_matrix()
# chair = class_chair.drop('class', 1).as_matrix()
# column = class_column.drop('class', 1).as_matrix()
# door = class_door.drop('class', 1).as_matrix()
# floor = class_floor.drop('class', 1).as_matrix()
# occlusion = class_occlusion.drop('class', 1).as_matrix()
# table = class_table.drop('class', 1).as_matrix()
# wall = class_wall.drop('class', 1).as_matrix()
# window = class_window.drop('class', 1).as_matrix()
#
# print(beam)
# print(ceiling)
#
#
# def create_ply(data, name):
#     outputFile = 'C:/Users/s143873/Desktop/Segmented point cloud/19+20/predicted2/pred_{}.ply'.format(name)
#     lines = [" ".join([ str(v) for v in i ]) + "\n" for i in data]
#     f = open(outputFile, 'w')
#     f.writelines("ply\n")
#     f.writelines("format ascii 1.0\n")
#     f.writelines("comment Created by Guido: use for blender\n")
#     f.writelines("element vertex {}\n".format(len(lines)))
#     f.writelines("property float x\n"
#                  "property float y\n"
#                  "property float z\n")
#     f.writelines("end_header\n")
#     f.writelines(lines)
#     f.close()
#
# create_ply(beam, 'beam')
# create_ply(ceiling, 'ceiling')
# create_ply(chair, 'chair')
# create_ply(column, 'column')
# create_ply(door, 'door')
# create_ply(floor, 'floor')
# create_ply(occlusion, 'occlusion')
# create_ply(table, 'table')
# create_ply(wall, 'wall')
# create_ply(window, 'window')
#
#
#
# # Write to .ply
#


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


lable_names = [
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
# print(classification_report(np.argmax(Y, axis=1), y_pred, target_names=lable_names))
cnf_matrix = (confusion_matrix(Y_test, predictions))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names,
                      title='Confusion matrix, without normalization')
plt.savefig('13_conf_matrix.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('13_conf_matrix2.png')