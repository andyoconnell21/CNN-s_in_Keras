import tensorflow as tf
from tensorflow import keras
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    new_train_images = np.expand_dims(train_images, axis=3)
    new_test_images = np.expand_dims(test_images, axis=3)

    if training:
        return (np.array(new_train_images), np.array(train_labels))
    else:
        return (new_test_images, test_labels)

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, train_img, train_lab, test_img, test_lab, T):
    train_labels = keras.utils.to_categorical(train_lab)
    test_labels = keras.utils.to_categorical(test_lab)

    model.fit(train_img, train_labels, validation_data = (test_img, test_labels), epochs=T)

def predict_label(model, images, index):
    prediction = model.predict(images)[index]
    value_list = []

    for x in range(len(prediction)):
        temp = (class_names[x], prediction[x])
        value_list.append(temp)

    value_list.sort(key = lambda x:x[1], reverse = True)

    print(value_list[0][0], ": ", round(value_list[0][1] * 100,2), "%")
    print(value_list[1][0], ": ", round(value_list[1][1] * 100,2), "%")
    print(value_list[2][0], ": ", round(value_list[2][1] * 100,2), "%")


