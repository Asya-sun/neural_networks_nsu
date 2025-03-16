import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def entropy(y):
    # Вычисляем вероятности классов
    class_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    # Вычисляем энтропию
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Добавляем маленькое значение для избежания логарифма нуля
    return entropy_value

def information_gain(X, y, feature_index):
    # Энтропия до разбиения
    total_entropy = entropy(y)

    # Получаем уникальные значения признака
    feature_values = np.unique(X[:, feature_index])

    # Вычисляем средневзвешенную энтропию после разбиения
    weighted_entropy = 0
    for value in feature_values:
        # Выбираем подмножество данных, где признак равен value
        subset_indices = X[:, feature_index] == value
        subset_y = y[subset_indices]

        # Вычисляем энтропию для подмножества
        subset_entropy = entropy(subset_y)

        # Вес подмножества (доля данных с этим значением признака)
        weight = len(subset_y) / len(y)

        # Добавляем вклад в средневзвешенную энтропию
        weighted_entropy += weight * subset_entropy

    # Information Gain
    IG = total_entropy - weighted_entropy
    return IG

def split_information(X, feature_index):
    # Получаем уникальные значения признака
    feature_values, counts = np.unique(X[:, feature_index], return_counts=True)

    # Вычисляем Split Information
    probs = counts / len(X)
    SI = -np.sum(probs * np.log2(probs + 1e-10))  # Добавляем маленькое значение для избежания логарифма нуля
    return SI


def gain_ratio(X, y, feature_index):
    # feature_index - индекс признака
    IG = information_gain(X, y, feature_index)
    SI = split_information(X, feature_index)

    GR = IG / SI if SI != 0 else 0
    return GR


def print_sorted_values(sorted_gain_ratios, n, metrics_name):
    if n <= 0:
        print("wrong number of features: ", n, "so i print 10")
        n = 10
    # Определяем ширину колонок для красиво форматированного вывода
    feature_width = max(len(feature) for feature, _ in sorted_gain_ratios) + 2
    metrics_width = 10  # например, 10 символов для выводимого значения

    # Заголовок таблицы
    print(f"{'Feature':<{feature_width}}{metrics_name:<{metrics_width}}")
    print("-" * (feature_width + metrics_width))

    # Выводим первые n признаков с форматированием
    for feature, ratio in sorted_gain_ratios[:n]:
        print(f"{feature:<{feature_width}}{ratio:<{metrics_width}.4f}")

def main():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Загрузка данных
    data = pd.read_csv('..\\data\\adult.data', names=names, index_col=False)

    data.replace(' ?', pd.NA, inplace=True)
    data.drop_duplicates(inplace=True)

    # Преобразуем столбцы с текстовыми значениями в категориальные
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # преобразуем последний столбец в булевый
    data['income'] = data['income'].map({' <=50K': False, ' >50K': True})
    data['income'] = data['income'].astype(bool)


    for column_name, column_data in data.items():
        if column_data.isnull().sum() > 0:
            if column_data.dtype in ['int64', 'float64']:
                column_data.fillna(column_data.mean(), inplace=True)
            else:
                column_data.fillna(column_data.mode()[0], inplace=True)

    # Преобразование категориальных признаков в фиктивные переменные
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


    features = data.columns

    categorical_from = data.select_dtypes(include=["bool"]).columns.drop("income")
    print(list(categorical_from))
    print("ALL TYPES")

    # Строим тепловую матрицу корреляции
    corr_matrix = data[features].corr()

    # Визуализация тепловой матрицы
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, cbar=True)
    plt.title('Тепловая матрица корреляции')
    plt.show()

    # Порог для сильной корреляции
    strong_corr_threshold = 0.8
    strong_corr_pairs = []

    # Ищем пары с сильной корреляцией
    strong_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > strong_corr_threshold:
                # Исключаем целевую переменную
                if corr_matrix.columns[i] != "income" and corr_matrix.columns[j] != "income":
                    strong_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    # Выводим сильно коррелирующие пары
    if strong_corr_pairs:
        print("Сильно коррелирующие пары:")
        for pair in strong_corr_pairs:
            print(f"{pair[0]} и {pair[1]}: {pair[2]}")
    else:
        print("Сильно коррелирующих признаков нет.")

    # Удаляем один из признаков в каждой паре
    features_to_drop = set()  # Множество для хранения признаков, которые нужно удалить
    for pair in strong_corr_pairs:
        feature_a, feature_b, corr_value = pair

        # Выбираем признак для удаления (например, тот, который имеет меньшую корреляцию с целевой переменной)
        if abs(corr_matrix.loc[feature_a, "income"]) < abs(corr_matrix.loc[feature_b, "income"]):
            features_to_drop.add(feature_a)
        else:
            features_to_drop.add(feature_b)

    if features_to_drop:
        # Удаляем выбранные признаки
        data = data.drop(columns=features_to_drop)
        print(f"Удалены признаки: {', '.join(features_to_drop)}")


    bin_number = 10

    # выбираем числовые признаки
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns    

    for feature in numerical_features:
        # Разбиваем на интервалы и получаем границы
        data[feature + '_group'], bins = pd.cut(data[feature], bins=bin_number, retbins=True)
        # Создаем метки для интервалов
        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
        # Присваиваем метки
        data[feature + '_group'] = pd.cut(data[feature], bins=bin_number, labels=labels)
        data.drop(feature, axis=1, inplace=True)


    # Добавляем новые категоризованные столбцы
    categorical_columns = [col for col in data.columns if '_group' in col]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


    # Преобразуем данные в numpy массивы для ручной реализации
    X = data.drop('income', axis=1).to_numpy()
    y = data['income'].to_numpy()

    feature_names = data.drop('income', axis=1).columns.tolist()
    information_gains = []
    for i in range(X.shape[1]):
        gr = information_gain(X, y, i)
        information_gains.append((feature_names[i], gr))


    sorted_information_gains = sorted(information_gains, key=lambda x: x[1], reverse=True)

    print_sorted_values(sorted_information_gains, 10, "INFORMATION GAINS")


    feature_names = data.drop('income', axis=1).columns.tolist()
    gain_ratios = []
    for i in range(X.shape[1]):
        gr = gain_ratio(X, y, i)
        gain_ratios.append((feature_names[i], gr))

    sorted_gain_ratios = sorted(gain_ratios, key=lambda x: x[1], reverse=True)

    print_sorted_values(sorted_gain_ratios, 10, "GAIN RATIOS")

if __name__ == "__main__":
    main()