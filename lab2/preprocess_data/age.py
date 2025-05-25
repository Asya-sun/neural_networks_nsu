import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

path_to_data = "D:\\Documents\\tpns\\laba_1\\data"
name = "age.csv"

def main():
    # names названия атрибутов
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Загрузка данных
    # параметр names нужен тк в data нет названия столбцов, и names - это как раз эти названия столбцов
    data = pd.read_csv(path_to_data + '\\adult.data', names=names, index_col=False)

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

    data.to_csv(name, index=False)

    # Определяем целевую переменную и признаки
    X = data.drop('age', axis=1)
    y = data['age']

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