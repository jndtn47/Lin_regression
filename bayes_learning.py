#https://habr.com/ru/post/329334/
##Алгоритм метод Наивного Байеса - два класса
#Разработчик - Самойлова
#Импорт  библиотек
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split # pip install scikit-learn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
# Загрузка и просмотр данных
#Поля:RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,
# Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
dataframe = pd.read_csv("C:\\tmp\\Churn_Modelling.csv")
dataframe.head()
#Преобразование данных
dataframe['Geography'].replace("France",1,inplace= True)
dataframe['Geography'].replace("Spain",2,inplace = True)
dataframe['Geography'].replace("Germany",3,inplace=True)
dataframe['Gender'].replace("Female",0,inplace = True)
dataframe['Gender'].replace("Male",1,inplace=True)
#Кросс валидация
array = dataframe.values
X = array[:, [4,6,8]] # столбцы 4,6,8 - Geography, Age, Balance,
print(X)
Y = array[:, 13]  # все значения 14-го столбца
Y = Y.astype('int')  #чтобы не было ошибки в данных
print(Y)
# Для избежания проблем с переобучением разделим наш набор данных:
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.4, random_state=0)
#Прогноз
model = GaussianNB()
model.fit(X_train ,y_train)  #ОБУЧЕНИЕ модели
# Точность предсказания
score = model.score(X_test, y_test)
print('Точность предсказания составила: ', score)
# save the model to disk
filename = 'C:\\tmp\\finalized_modelNB.sav'
pickle.dump(model, open(filename, 'wb')) #Сохранение модели
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# прогноз для Самойловой. Она в этом банке
x_test = np.array([[3,59,115046.74]])  # это Самойлова, она из Германии, 59 лет, счет 115046.74 $
predicted = model.predict(x_test)
print('Прогноз ухода Морозова из банка = ',predicted)
