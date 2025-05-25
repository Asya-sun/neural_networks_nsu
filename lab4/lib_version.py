import os
# Отключаем GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.api.datasets import mnist


# Предобработка данных
def preprocess_data(train_images, train_labels, test_images, test_labels):
    # Нормализация и добавление размерности канала
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # Преобразование меток в one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

# Создание модели LeNet-5
def create_lenet5():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def main():
    # Загрузка данных    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Предобработка
    train_images, train_labels, test_images, test_labels = preprocess_data(
        train_images, train_labels, test_images, test_labels
    )
    
    # Создаём оптимизатор с нужным learning rate
    optimizer = Adam(learning_rate=0.01)

    # Создание и обучение модели
    model = create_lenet5()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
    
    # Оценка модели
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Получаем предсказания для всех тестовых данных
    test_pred_probs = model.predict(test_images)
    test_pred_classes = np.argmax(test_pred_probs, axis=1)
    test_true_classes = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(test_true_classes, test_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("lib_cm.png")
    plt.close()
    print("Confusion matrix - 'lib_cm.png'")


if __name__ == "__main__":
    main()