import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error

from sklearn.preprocessing import StandardScaler

path_to_data = "D:\\Documents\\tpns\\laba_2\\preprocess_data\\age.csv"

class RegressorMLP(object):
	# функция активации - сигмоида на скрытых, линейная активация на выходом
	# функция потерь - mse
	def __init__(self, layers, random_state=None):
		np.random.seed(random_state)
		self.num_layers = len(layers)
		self.layers = layers
		self.initialize_weights()

	def initialize_weights(self):
		self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]

	def fit(self, training_data, epochs=500, eta=0.001, batch_size=1, test_size=0.2):
		n = len(training_data)
		test_size = int(n * test_size)

		for epoch in range(epochs):
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
			for mini_batch in mini_batches:
				self.batch_update( mini_batch, eta)

	def batch_update(self, mini_batch, eta):
		# Собираем батч в матрицы
		# Исправляем транспонирование
		X_batch = np.array([x for x, y in mini_batch])  # Форма (batch_size, n_features)
		y_batch = np.array([y for x, y in mini_batch])

		delta_nabla_b, delta_nabla_w = self.back_propogation(X_batch.T, y_batch)  # Транспонируем здесь

		self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, delta_nabla_w)]
		self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, delta_nabla_b)]

	def back_propogation(self, X_batch, y_batch):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# Прямое распространение для батча
		activation = X_batch
		activations = [activation]
		zs = []
		

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self.sigmoid(z) if w is not self.weights[-1] else z
			activations.append(activation)

		# Обратное распространение
		delta = (activations[-1] - y_batch)  # Для MSE
		nabla_b[-1] = delta.sum(axis=1, keepdims=True)
		nabla_w[-1] = np.dot(delta, activations[-2].T)
		
		for l in range(2, self.num_layers):
			sp = self.derivative(zs[-l])
			delta = np.dot(self.weights[-l+1].T, delta) * sp
			nabla_b[-l] = delta.sum(axis=1, keepdims=True)
			nabla_w[-l] = np.dot(delta, activations[-l-1].T)

		return nabla_b, nabla_w


	@staticmethod
	def delta(a, y):
		# потому что функция потреь - среднеквадратическая ошибка и на выходном слое используется линейная активация
		return (a-y)


	@staticmethod
	def sigmoid(z):
		return expit(z)  # expit автоматически работает с массивами

	@staticmethod
	def derivative(z):
		s = expit(z)
		return s * (1 - s)
		
	def predict(self, X):
		"""
		Векторизованное предсказание для входных данных X
		X: массив формы (n_samples, n_features)
		Возвращает массив предсказаний формы (n_samples,)
		"""
		# Транспонируем X для матричных операций (n_features, n_samples)
		activation = X.T

		for i, (b, w) in enumerate(zip(self.biases, self.weights)):
			z = np.dot(w, activation) + b
			# Для последнего слоя линейная активация
			activation = z if i == len(self.weights)-1 else self.sigmoid(z)

		return activation.flatten()
	

def main():
	data = pd.read_csv(path_to_data)

	# Предположим, что целевой признак называется 'age'
	target_column = 'age'

	# Разделение данных на признаки (X) и целевую переменную (y)
	X = data.drop(columns=[target_column]).values  # Признаки
	y = data[target_column].values  # Целевая переменная	

    # Разделение данных на обучающую и тестовую выборки
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	# Создание и обучение модели
	# Пример архитектуры: [количество входных признаков, 100 нейронов в скрытом слое, 1 выходной нейрон]
	input_size = X_train.shape[1]
	layers = [input_size, 50, 1]  # Архитектура сети

	mlp = RegressorMLP(layers, random_state=42)

	# Преобразуем данные в формат, подходящий для обучения (список кортежей (x, y))
	training_data = list(zip(X_train, y_train))
	
	print(f"Input size: {input_size}")  # Должно совпадать с X_train.shape[1]
	print(f"Layers: {layers}")         # Например: [97, 50, 1]

	epochs_number = 100
	learning_rate = 0.01
	minibatches = 32

	print(f"epochs_number : {epochs_number}\nlearning_rate : {learning_rate}\nminibatches : {minibatches}\n")

	# Обучение модели
	mlp.fit(training_data, epochs=epochs_number, eta=learning_rate, batch_size=minibatches)

	# Предсказание на тестовых данных
	y_pred = mlp.predict(X_test)

	# Оценка модели
	mae = mean_absolute_error(y_test, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	mape = mean_absolute_percentage_error(y_test, y_pred)
	msle = mean_squared_log_error(y_test, y_pred)

	r2 = r2_score(y_test, y_pred)

	print(f"Mean Absolute Error (MAE): {mae:.4f}")
	print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
	print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
	print(f"Mean Squared Logarithmic Error (MSLE): {msle:.4f}")
	print(f"R^2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()