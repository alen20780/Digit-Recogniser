import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def get_mnist_data():
    path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return (x_train, y_train, x_test, y_test)

def train_model(x_train, y_train):
    x_train = x_train.reshape(-1, 28*28) / 255.0
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    
    return knn, scaler

def predict(knn, scaler, img):
    img = img.reshape(1, -1) / 255.0
    img = scaler.transform(img)
    res = knn.predict(img)
    return str(res[0])