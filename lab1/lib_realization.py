import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

def main():

    # names названия атрибутов
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Загрузка данных
    # параметр names нужен тк в data нет названия столбцов, и names - это как раз эти названия столбцов
    data = pd.read_csv('..\\data\\adult.data', names=names, index_col=False)

    # Заменяем пропуски на NaN
    data.replace(' ?', pd.NA, inplace=True) 
    # Удаляем дубликаты
    data.drop_duplicates(inplace=True) 

    # Преобразуем столбцы с текстовыми значениями в категориальные
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # преобразуем последний столбец в булевый
    data['income'] = data['income'].map({' <=50K': False, ' >50K': True})
    data['income'] = data['income'].astype(bool)
    # # преобразуем последний столбец в int
    # data['income'] = data['income'].replace({' <=50K': 0, ' >50K': 1})
    # data['income'] = data['income'].astype(int)

    # Заполняем отсутствующие значения
    for column_name, column_data in data.items():
        if column_data.isnull().sum() > 0:
            # Если столбец числовой, заполняем средним значением
            if column_data.dtype in ['int64', 'float64']:
                column_data.fillna(column_data.mean(), inplace=True)
            # Если столбец категориальный, заполняем наиболее частым значением
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

    plt.figure(figsize=(15, 10))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
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


    # print(list(data.columns))
    # print("ALL TYPES")

    # Определяем целевую переменную и признаки
    X = data.drop('income', axis=1)
    y = data['income']

    # Обучаем дерево решений с использованием критерия entropy (information gain)
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X, y)

    # Получаем важность признаков
    # feature importance - это не information gain, а вообще друга метрика
    # information gain не нормализована, а feature importance нормализована
    importances = tree.feature_importances_

    # Создаем DataFrame для визуализации важности признаков
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print("FEATURE IMPORTANCES")
    print(feature_importance.head(10))

    # тут используем еще одну метрику - чисто для сравнения
    # Mutual Information/Взаимная информация — это мера того, насколько сильно признаки и целевая переменная зависят друг от друга.
    mi_scores = mutual_info_classif(X, y)
    mi_scores_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    mi_scores_df = mi_scores_df.sort_values(by='mi_score', ascending=False)

    print("MUTUAL INFORMATION")
    print(mi_scores_df[:10])


# __name__
if __name__=="__main__":
    main()