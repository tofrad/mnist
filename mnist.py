import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

layers = tf.keras.layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)  # SHAPE: (pics, rows, pixel)
print(y_train)  # SHAPE: result

#normalize
x_train = x_train / 255.
x_test = x_test / 255.

example_image = x_test[2]

#model with fcl
model2 = tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#model_cnn
model = tf.keras.models.Sequential([ #28x28
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(16, 3, activation='relu'), #28x28x3
    layers.Conv2D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

savepoint = "model.h5"
checkpoint = ModelCheckpoint(savepoint,
                             monitor= 'accuracy',
                             save_best_only=True,
                             verbose=1)

# train params
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)

#train
history = model.fit(x_train, y_train,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    callbacks=[checkpoint])

#plot trianing history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

#test the model
model.evaluate(x_test, y_test)

#test with one example image
np.reshape(example_image, (1, 28, 28))
prediction = model.predict(np.reshape(example_image, (1, 28, 28)))

plt.imshow(np.reshape(x_test[2], [28,28]), cmap='gray')
plt.show()

print("Prediction: ", np.argmax(prediction))

