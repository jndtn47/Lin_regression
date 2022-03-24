import pandas as pd
# Загрузка и просмотр данных
# Loading the dataset into our dataframe
df = pd.read_csv("C:\\tmp\\moscow_dataset_2020.csv", delimiter=",")
# Create features – 6 column
features = ["floorNumber","floorsTotal","totalArea","kitchenArea", "latitude", "longitude",] #Этаж, Число этажей
X = df[features]
print(X)
y = df['price']
print(y)
