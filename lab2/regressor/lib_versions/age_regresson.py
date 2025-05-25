import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

path_to_data = "D:\\Documents\\tpns\\laba_2\\preprocess_data\\age.csv"

def main():
    # Загрузка данных
    data = pd.read_csv(path_to_data)

    # Предобработка данных
    X = data.drop(columns=['age'])  # Признаки
    y = data['age']  # Целевая переменная

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_neurons = 50
    num_layers = 1
    learn_rate = 0.01
    epochs_number = 10    
    batch_size = 32    
    print(f" len(data){ len(data)}")
    num_batches = len(data) / batch_size
    num_iterations_per_epoch = len(data) / batch_size
    total_iterations = int(num_iterations_per_epoch * epochs_number)

    print(f"num_neurons: {num_neurons} num_layers: {num_layers} epochs_number: {epochs_number} learn_rate: {learn_rate} batch_size: {batch_size}")



    # Создание и обучение модели
    # Используем MLPRegressor (многослойный перцептрон)
    model = Pipeline([
        ('scaler', StandardScaler()),  # Масштабирование данных
        ('mlp', MLPRegressor(
            # (100, 50)
            hidden_layer_sizes=tuple([num_neurons] * num_layers), 
            activation='relu',  # Функция активации
            solver='adam',  # Алгоритм оптимизации
            max_iter=total_iterations,  # Максимальное количество итераций
            random_state=42,
            learning_rate_init=learn_rate,
            batch_size=batch_size
        ))
    ])


    model.fit(X_train, y_train)

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)

    # Оценка модели
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Squared Logarithmic Error (MSLE): {msle:.4f}")
    print(f"R^2 Score: {r2}")


# __name__
if __name__=="__main__":
    main()