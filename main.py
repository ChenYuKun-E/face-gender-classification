import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from data import Data,IMAGE_W,IMAGE_H

print(tf.__version__)

data = Data()
train_data = data.get_train_data(-9)
train_x = train_data['xs'] / 255
train_y = train_data['labels']

test_x = train_data['validation_xs'] / 255
test_y = train_data['validation_labels']

my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, min_delta=0, monitor='val_loss')]

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(IMAGE_W, IMAGE_H, 1), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=train_x,
          y=train_y,
          batch_size=32,
          epochs=30,
          verbose=1,
          callbacks=my_callbacks,
          validation_split=0.05,
          shuffle=True
          )

predictions = model.predict(test_x)

class_names = ["Female", "Male"]

plt.figure(figsize=(12, 6))
for i in range(min(9, len(test_y))):
    result = predictions[i]
    max_label = int(np.argmax(result))
    correct_label = int(np.argmax(test_y[i]))

    plt.subplot(3, 6, 2 * i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = test_x.reshape(test_x.shape[0], IMAGE_W, IMAGE_H)[i]
    plt.imshow(img, cmap="gray")
    plt.xlabel("{} - prob:{:2.0f}%".format(class_names[max_label], 100 * np.max(result)))

    plt.subplot(3, 6, 2 * i + 2)
    plt.grid(False)
    plt.yticks([])
    plt.ylim([0, 1])
    bar = plt.bar(range(2), result)
    bar[max_label].set_color('red')
    bar[correct_label].set_color('green')

plt.show()
