#Импорт  библиотек
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
# Загрузка и просмотр данных
# Loading the dataset into our dataframe
df = pd.read_csv("C:\\tmp\\moscow_dataset_2020.csv", delimiter=",")
# Create features
features = ["floorNumber","floorsTotal","totalArea","kitchenArea", "latitude", "longitude",] #Этаж, Число этажей
X = df[features]
print(X)
y = df['price']
print(y)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Initiate mode
model = LinearRegression()
# Fit model
model.fit(X_train, y_train)
# Get predictions
test_predict = model.predict(X_test)# предсказываем цену квартиры на тестовых данных
train_predict = model.predict(X_train)# предсказываем цену квартиры на обучающих данных
# Score and compare
#метрики оценки качества
print('коэффициент детерминации для теста=',r2_score(y_test, test_predict)) # сравниваем истинные и предсказанные данные
print('коэффициент детерминации= для обучения=',r2_score(y_train, train_predict))  # сравниваем истинные и предсказанные данные
plt.scatter(test_predict, y_test) # сравниваем истинные и предсказанные данные для тестовой выборки
plt.plot(test_predict, model.predict(X_test), color='red', linewidth=2);
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
#Оценки точности
#среднеквадратическая ошибка (RMSE)
mse = metrics.mean_squared_error(y_test, test_predict)
# m is the number of training examples
m = 31192
rmse = np.sqrt(mse/m)
print('RMSE=',rmse)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# some time later...
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
 # прогноз для Морозова. Вот ее квартира:
X_test = np.array([[3, 10.0, 55.0, 6.0, 55.486698,37.595321000000006, ]]) #55.786698,
#Предскажем ей цену
test_predict = loaded_model.predict(X_test)# предсказываем цену квартиры Морозова
print('Предскажем!!!Цена квартиры для Морозова в рублях =',test_predict)
