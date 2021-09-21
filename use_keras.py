import random  # for generating random numbers

import matplotlib.pyplot as plt  # MATLAB like plotting routines
# The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
import numpy as np
from keras.layers.core import Dense, Dropout, Activation  # Types of layers to be used in our model
from keras.models import Sequential  # Model type to be used
from tensorflow.python.keras.utils import np_utils

from util import get_music_files, load_data

files = get_music_files()
data_n = 200
ts_n = 30
(X_train, y_train), (X_test, y_test) = load_data(files, data_n, ts_n)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

plt.rcParams['figure.figsize'] = (9, 9)  # Make the figures a bit bigger
xsz, ysz = X_train[0].shape

for i in range(9):
    plt.subplot(3, 3, i + 1)
    num = random.randint(0, len(X_train) - 1)
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))

plt.tight_layout()
plt.show()


# just a little function for pretty printing a matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


# now print!
matprint(X_train[num])

X_train = X_train.reshape(data_n, xsz * ysz)  # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
X_test = X_test.reshape(ts_n, xsz * ysz)  # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

X_train = X_train.astype('float32')  # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')
# X_train /= 255  # normalize each value for each pixel for the entire vector for each input
# X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
print("Y_train", y_train)

nb_classes = 4  # number of unique digits

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(512, input_shape=(xsz * ysz,)))

model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128, epochs=10,
          verbose=1)

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predict_x = model.predict(X_test)
predicted_classes = np.argmax(predict_x, axis=1)
# predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]

incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[correct].reshape(xsz, ysz), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.tight_layout()
plt.show()

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[incorrect].reshape(xsz, ysz), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

plt.tight_layout()
plt.show()
