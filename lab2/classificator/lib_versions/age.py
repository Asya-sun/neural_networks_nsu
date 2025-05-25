import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

path_to_data = "D:\\Documents\\tpns\\laba_2\\preprocess_data\\age.csv"



def main():
	data = pd.read_csv(path_to_data,  index_col=False)
      
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
      
	data, labels = categorize_age(data, bin_number=5)
	# bins = [0, 18, 30, 45, 60, np.inf]  # Границы категорий
	# labels = ['child', 'young', 'adult', 'middle-aged', 'senior']
	# # labels = ['0-18', '18-30', '30-45', '45-60', '60+']
	# data['age_category'] = pd.cut(data['age'], bins=bins, labels=labels)

	# Подготавливаем данные
	X = data.drop(columns=['age', 'age_category'])
	y = data['age_category']

	# Кодируем метки
	# le = LabelEncoder()
	# y_encoded = le.fit_transform(y)

	ohe = OneHotEncoder(sparse_output=False)
	y_encoded = ohe.fit_transform(y.values.reshape(-1, 1))

	# Разделяем данные
	X_train, X_test, y_train, y_test = train_test_split(
		X, y_encoded, 
		test_size=0.2, 
		stratify=y_encoded,  # Для сохранения баланса классов
		random_state=42
	)

	# Масштабируем признаки
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	num_neurons = 50
	num_layers = 10
	max_iters = 1000
	learn_rate = 0.01
	print(f"num_neurons: {num_neurons} num_layers: {num_layers} max_iters: {max_iters} learn_rate: {learn_rate}")


	# Обучаем модель
	mlp = MLPClassifier(
		hidden_layer_sizes=([num_neurons] * num_layers), 
		activation='relu',             # Функция активации
		solver='adam',                # Алгоритм оптимизации
		learning_rate_init=learn_rate,
		max_iter=max_iters,
    	early_stopping=True,         # Ранняя остановка
		random_state=42
	)

	mlp.fit(X_train_scaled, y_train)

	# Оцениваем
	y_pred = mlp.predict(X_test_scaled)
	
	y_pred_labels = np.argmax(y_pred, axis=1)
	y_test_labels = np.argmax(y_test, axis=1)

	print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
	print("\nClassification Report:")
	print(classification_report(y_test, y_pred, target_names=labels))

	# Визуализация матрицы ошибок
	cm = confusion_matrix(y_test_labels, y_pred_labels)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	disp.plot(cmap='Blues')
	plt.show()



# __name__
if __name__=="__main__":
    main()