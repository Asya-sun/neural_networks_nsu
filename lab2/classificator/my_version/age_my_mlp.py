import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

path_to_data = "D:\\Documents\\tpns\\laba_2\\preprocess_data\\age.csv"

class ClassificatorMLP(object):
    # функция активации - сигмоида в скрытых слоях, softmax в выходных слоях
    # функция потерь - категориальная кросс-энтропия
    def __init__(self, layers, num_classes, random_state=None):
        np.random.seed(random_state)
        self.num_layers = len(layers)
        self.layers = layers
        self.num_classes = num_classes
        self.initialize_weights()

    def initialize_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # Корректируем выходной слой
        self.biases[-1] = np.random.randn(self.num_classes, 1)
        self.weights[-1] = np.random.randn(self.num_classes, self.layers[-2]) / np.sqrt(self.layers[-2])



    def fit(self, training_data, epochs=500, eta=0.001, batch_size=32):
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]

            for mini_batch in mini_batches:
                self.batch_update( mini_batch, eta)

    def batch_update(self, mini_batch, eta):
        X_batch = np.array([x for x, y in mini_batch])
        Y_batch = np.array([y for x, y in mini_batch])
        
        nabla_b, nabla_w = self.back_propagation(X_batch.T, Y_batch)
        
        self.weights = [w - (eta / len(mini_batch)) * nw  for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_propagation(self, X_batch, Y_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Прямое распространение для батча
        activation = X_batch  # (features, batch_size)
        activations = [activation]
        zs = []
        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Выходной слой
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        a = self.softmax(z)
        activations.append(a)
        zs.append(z)
        
        delta = (activations[-1] - Y_batch.T)  # (classes, batch_size)        
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T) 
        # print(Y_batch.shape)
        
        for l in range(2, self.num_layers):
            sp = self.derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_b, nabla_w

    
    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    @staticmethod
    def sigmoid(z):
        return expit(z)

    @staticmethod
    def derivative(z):
        return ClassificatorMLP.sigmoid(z) * (1 - ClassificatorMLP.sigmoid(z))

    
    def predict(self, X):
        # Векторизованная версия для всего набора данных
        X = X.T  # (features, samples)
        for i, (b, w) in enumerate(zip(self.biases[:-1], self.weights[:-1])):
            X = self.sigmoid(np.dot(w, X) + b)
        # Выходной слой
        z = np.dot(self.weights[-1], X) + self.biases[-1]
        return np.argmax(self.softmax(z), axis=0)









def main():
    # Загружаем данные
    data = pd.read_csv(path_to_data)
    
    def categorize_age(data, age_column='age', bin_number=10, new_name = 'age_category'):
        # Находим минимальный и максимальный возраст
        min_age = data[age_column].min()
        max_age = data[age_column].max()

        # Создаем равномерные интервалы
        bins = np.linspace(min_age, max_age, bin_number + 1)

        # Округляем границы до целых чисел
        bins = np.unique(bins.round().astype(int))

        # Создаем подписи для категорий
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

        # Добавляем категории в DataFrame
        data[new_name] = pd.cut(
            data[age_column], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )

        return data,labels

    data, age_labels = categorize_age(data, bin_number=5)
    
    # Подготавливаем данные
    X = data.drop(columns=['age', 'age_category']).values
    y = data['age_category'].values
    
    # Кодируем метки
    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y.reshape(-1, 1))
    num_classes = y_encoded.shape[1]
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape)
    print(X_train.shape[0])
    
    # Параметры модели
    input_size = X_train.shape[1]
    layers = [input_size, 100, 50, num_classes]  # Архитектура сети
    
    # Подготавливаем данные для обучения
    training_data = list(zip(X_train, y_train))
    
    # Обучаем модель
    mlp = ClassificatorMLP(
        layers=layers,
        num_classes=num_classes,
        random_state=42
    )
    
    mlp.fit(
        training_data=list(zip(X_train, y_train)),
        epochs=50,
        eta=0.01,
        batch_size=32
    )
    
    # Оцениваем
    y_pred = mlp.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=age_labels))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=age_labels)
    disp.plot(cmap='Blues')
    plt.show()
    

if __name__ == "__main__":
    main()