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

seed = 7
np.random.seed(seed)

# load
dataset = np.load('training.npy')
testset = np.load('testset.npy')



lblnms = [
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

lbls = dataset[:,8]
msk = np.logical_or(lbls == lblnms.index("wall"), lbls == lblnms.index("floor"))

print(dataset[0:10])
dataset = dataset[msk]
print(dataset[0:10])

xs, ys, lbls = dataset[:,5][::100], dataset[:,6][::100], dataset[:,8][::100]
mp = plt.get_cmap("tab10")

for i in range(np.amax(lbls) + 1):
    mask = lbls == i
    plt.scatter(xs[mask], ys[mask], c=mp(i), s=1, label=lblnms[i])
plt.legend()
plt.show()
plt.clf()
quit()
X_orient = dataset[:,7]

encoder = LabelEncoder()
encoder.fit(X_orient)
encoded_X_orient = encoder.transform(X_orient)
dummy_X_orient = np_utils.to_categorical(encoded_X_orient)

test_orient = testset[:,7]
test_class = testset[:,8]

encoder = LabelEncoder()
encoder.fit(test_orient)
encoded_test_orient = encoder.transform(test_orient)
dummy_test_orient = np_utils.to_categorical(encoded_test_orient)

encoder.fit(test_class)
encoded_test_class = encoder.transform(test_class)
dummy_test_class = np_utils.to_categorical(encoded_test_class)




print(dataset[:,3:7].shape)
print(testset[:,3:7].shape)
# print(dummy_X_orient)

new_training = np.concatenate((dataset[:,3:7], dummy_X_orient), axis=1)
new_testset = np.concatenate((testset[:,3:7], dummy_test_orient, dummy_test_class), axis=1)
# print(new_training.shape)

# split into input (X) and output (Y) variables
X = new_training
# print(X)
Y = dataset[:,8]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert to dummies
dummy_y = np_utils.to_categorical(encoded_Y)

print(Y[0:10])
print(dummy_y[:,0:10])

#
#

# print(dummy_y)

# create model ---------------------------------------------------------------------------------------------------------
# 7 inputs -> [hidden layers -> 10 outputs
model = Sequential()
model.add(Dense(7, input_dim=7, activation='relu'))
#model.add(Dense(14, activation='relu'))
#model.add(Dense(14, activation='relu'))
model.add(Dense(dummy_y.shape[1], activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae', 'accuracy'])

# fit
history = model.fit(X, dummy_y, validation_split=.15, epochs=2, batch_size=50000)

print(X[0:10])
values = model.predict(X)
print(values[0:10])
values2 = np.argmax(values)
print(values2[0:10])
print(Y[0:10])
print(sklearn.metrics.confusion_matrix(Y, values2))

# evaluate
# scores = model.evaluate(X, dummy_y, batch_size=10000)
# print(model.history.keys()
# estimator = KerasClassifier(build_fn=model, validation_split=0.12, epochs=2, batch_size=100000)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#summarize history for accuracy and loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'])
plt.show()

"""If the training loss goes under validation loss -> model overfitting!"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel(' and loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'])
plt.show()



model.save('neural_net_1.h5')

#
#
# scores = model.evaluate(X, dummy_y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# predictions = model.predict(X)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)