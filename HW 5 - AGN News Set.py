import pandas as pd
import keras

# Cardy Wei
# Professor Curro
# Deep Learning Assignment 5

max_len = 1012
num_classes = 4
epochs = 3
batch_size = 128

train = pd.read_csv('ag_news_csv/train.csv', names=["class", "title", "desc"])
test = pd.read_csv('ag_news_csv/test.csv', names=["class", "title", "desc"])

x_train = train["title"] + " " + train["desc"]
x_test = test["title"] + " " + test["desc"]

y_train = train["class"] - 1
y_test = test["class"] - 1

x_train = x_train[20000:]
y_train = y_train[20000:]
x_val = x_train[:20000]
y_val = y_train[:20000]

t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(x_train)

x_train = t.texts_to_sequences(x_train)
x_val = t.texts_to_sequences(x_val)
x_test = t.texts_to_sequences(x_test)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, padding="post", truncating="post", maxlen=max_len)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, padding="post", truncating="post", maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, padding="post", truncating="post", maxlen=max_len)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(keras.layers.Embedding(len(t.word_counts), 64, input_length=max_len))
model.add(keras.layers.MaxPooling1D(pool_size = 2))
model.add(keras.layers.Dropout(.4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# C:\Users\cardy\Desktop>python MLHW5.py
# Using TensorFlow backend.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 1012, 64)          4102272
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 506, 64)           0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 506, 64)           0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 32384)             0
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 129540
# =================================================================
# Total params: 4,231,812
# Trainable params: 4,231,812
# Non-trainable params: 0
# _________________________________________________________________
# Train on 100000 samples, validate on 20000 samples
# Epoch 1/3
# 2018-10-10 17:20:00.836661: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# 2018-10-10 17:20:01.478405: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1405] Found device 0 with properties:
# name: GeForce 940MX major: 5 minor: 0 memoryClockRate(GHz): 0.8605
# pciBusID: 0000:01:00.0
# totalMemory: 2.00GiB freeMemory: 1.66GiB
# 2018-10-10 17:20:01.486599: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1484] Adding visible gpu devices: 0
# 2018-10-10 17:20:02.951061: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-10-10 17:20:02.955509: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:971]      0
# 2018-10-10 17:20:02.960616: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:984] 0:   N
# 2018-10-10 17:20:02.965537: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1412 MB memory) -> physical GPU (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0, compute capability: 5.0)
# 100000/100000 [==============================] - 57s 565us/step - loss: 0.4873 - acc: 0.8383 - val_loss: 0.1987 - val_acc: 0.9401
# Epoch 2/3
# 100000/100000 [==============================] - 50s 502us/step - loss: 0.1835 - acc: 0.9407 - val_loss: 0.1204 - val_acc: 0.9645
# Epoch 3/3
# 100000/100000 [==============================] - 50s 501us/step - loss: 0.1216 - acc: 0.9608 - val_loss: 0.0715 - val_acc: 0.9804
# Test loss: 0.25114226884512525
# Test accuracy: 0.9206842105263158