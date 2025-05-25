import pandas as pd
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_curve, roc_auc_score, precision_recall_curve,
                             auc, average_precision_score, classification_report)

path = "D:\\Documents\\tpns\\laba_2\\preprocess_data\\processed_data.csv"


def main():
    data = pd.read_csv(path,  index_col=False)

    # попробовать взять другой параметр - возраст


    # Определяем целевую переменную и признаки
    X = data.drop('income', axis=1)
    y = data['income']

    # print(len(X.columns))

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабируем данные
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_neurons = 50
    num_layers = 10
    max_iters = 1000
    learn_rate = 0.001
    print(f"num_neurons: {num_neurons} num_layers: {num_layers} max_iters: {max_iters} learn_rate: {learn_rate}")

    # Создаем и обучаем MLP-классификатор
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=tuple([num_neurons] * num_layers),
        activation='relu',            # Функция активации: ReLU
        solver='adam',                # Алгоритм оптимизации: Adam
        max_iter=max_iters,                # Максимальное количество итераций
        random_state=42,
        learning_rate_init=learn_rate
    )

    labels = ['0-18', '18-30', '30-45', '45-60', '60+']
    mlp_classifier.fit(X_train, y_train)

    # Предсказания на тестовой выборке
    y_pred = mlp_classifier.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")
    print(precision)
    # True Positive Rate
    recall = recall_score(y_test, y_pred)
    print(f"TPR: {recall}")
    f1 = f1_score(y_test, y_pred)
    print(f"F1: {f1}")


    conf_matrix = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = conf_matrix.ravel()

    fpr = fp / (fp + tn)
    print(f"FPR: {fpr}")

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    

    


# __name__
if __name__=="__main__":
    main()