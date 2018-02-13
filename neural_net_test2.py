import numpy as np
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
from keras.utils import plot_model


seed = 7
np.random.seed(seed)

dataset = np.load('training_FPFH_height.npy')
print(dataset.shape)

fpfh_norm = np.divide(dataset[:,0:33], 100)
normals_curv = dataset[:,33:37]
height_norm = dataset[:, 41:43]
print(height_norm)

new_dataset = np.concatenate((fpfh_norm, normals_curv), axis=1)
dataframe = pd.DataFrame(dataset)
value_count = dataframe[37].value_counts()
print(value_count)



class_dict = {"beam":0, "ceiling":1, "chair":2, "column":3, "door":4, "floor":5, "occlusion":6, "table":7, "wall":8, "window":9}
class_weight = {0:max(value_count)/value_count[0],
                1:max(value_count)/value_count[1],
                2:max(value_count)/value_count[2],
                3:max(value_count)/value_count[3],
                4:max(value_count)/value_count[4],
                5:max(value_count)/value_count[5],
                6:max(value_count)/value_count[6],
                7:max(value_count)/value_count[7],
                8:max(value_count)/value_count[8],
                9:max(value_count)/value_count[9]
                }

#new_training = np.concatenate((dataset[:,0:33], dummy_X_orient), axis=1)
# X = new_training
X = new_dataset
Y = np_utils.to_categorical(dataset[:,37])

print(X, X.shape)
print(Y, Y.shape)

# create model ---------------------------------------------------------------------------------------------------------
# 7 inputs -> [hidden layers -> 10 outputs
model = Sequential()
model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
model.add(Dense(X.shape[1]*2, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.2))
# model.add(Dense(X.shape[1]*2, activation='relu'))
model.add(Dense(X.shape[1]*2, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax')) # activ = sigmoid gives probability table

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fit
history = model.fit(X, Y, validation_split=.25, epochs=30, batch_size=10000, verbose=2, class_weight=class_weight)
# model.fit(X, Y, validation_split=.33, epochs=50, batch_size=10000, verbose=2, class_weight=class_weight)

model.save('neural_net_FPFH.h5')

#summarize history for accuracy and loss
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'])
plt.savefig('history.png')

"""If the training loss goes under validation loss -> model overfitting!"""
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel(' and loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'])
plt.savefig('history2.png')

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

Y_pred = model.predict(X)
# print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
# print(y_pred)
#
y_pred = model.predict_classes(X)
# print(y_pred)
# p = model.predict_proba(X)

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
print(classification_report(np.argmax(Y, axis=1), y_pred, target_names=lable_names))
cnf_matrix = (confusion_matrix(np.argmax(Y, axis=1), y_pred))
print('%%%%%%%%%%%%%%%%%%%%%%')
print(np.argmax(Y, axis=1), y_pred)
print(Y)
print(y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names,
                      title='Confusion matrix, without normalization')
plt.savefig('conf_matrix.png')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('conf_matrix2.png')
