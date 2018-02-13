import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
import pandas as pd

seed = 7
np.random.seed(seed)

dataset = np.load('training.npy')

dataframe = pd.DataFrame(dataset)
value_count = dataframe[8].value_counts()

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

# OBSOLETE
# X_orient = dataset[:,7]
#
#
# dummy_X_orient = np_utils.to_categorical(dataset[:,7])
# new_training = np.concatenate((dataset[:,3:7], dummy_X_orient), axis=1)

# X = new_training
n_samples = round(len(dataset) / 100)
dataset1 = dataset[np.random.choice(dataset.shape[0],n_samples,replace=False),:]


X = dataset1[:,3:7]
Y = np_utils.to_categorical(dataset1[:,8])

print(X)
print(Y.shape[1])

# create model ---------------------------------------------------------------------------------------------------------
# 7 inputs -> [hidden layers -> 10 outputs
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(Y.shape[1], activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fit
history = model.fit(X, Y, validation_split=.15, epochs=50, batch_size=1000, verbose=2, class_weight=class_weight)

#summarize history for accuracy and loss
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'])


"""If the training loss goes under validation loss -> model overfitting!"""
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel(' and loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'])
plt.show()

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
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

y_pred = model.predict_classes(X)
print(y_pred)
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

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=lable_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



# evaluate
# scores = model.evaluate(X, dummy_y, batch_size=10000)
# print(model.history.keys()
# estimator = KerasClassifier(build_fn=model, validation_split=0.12, epochs=2, batch_size=100000)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



