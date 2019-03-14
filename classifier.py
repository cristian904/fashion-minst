import struct
import numpy as np 
from keras.utils import np_utils
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras.models import Sequential

def read_idx(filename):
    with open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
x_train = read_idx("./fashion_mnist/train-images-idx3-ubyte")
y_train = read_idx("./fashion_mnist/train-labels-idx1-ubyte")
x_test = read_idx("./fashion_mnist/t10k-images-idx3-ubyte")
y_test = read_idx("./fashion_mnist/t10k-labels-idx1-ubyte")

batch_size = 128
epochs = 3

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

#create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
            batch_size=batch_size, 
            epochs = epochs,
            shuffle = True,
            verbose = 1)

scores = model.evaluate(x_test, y_test, verbose=0)
print("FASHION-MINST test accuracy:", scores[1])