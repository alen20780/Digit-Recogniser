import tensorflow as tf
import numpy as np

def get_mnist_data():
    path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return (x_train, y_train, x_test, y_test)

def train_model(x_train, y_train, x_test, y_test):
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
                
    callbacks = MyCallback()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    print(history.epoch, history.history['accuracy'][-1])
    return model

def predict(model, img):
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    return str(tf.argmax(res, axis=1).numpy()[0])

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)