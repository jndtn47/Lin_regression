from django.shortcuts import render


def index(request):
    return render(request, "index.html")


import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from .forms import User_Haus_Form



def ML2(request):
    msg = 'Заполните поля'
    if request.method == "POST":
        form = User_Haus_Form(request.POST)
        if form.is_valid():
            floorNumber = form.cleaned_data.get("floorNumber")
            floorsTotal = form.cleaned_data.get("floorsTotal")
            totalArea = form.cleaned_data.get("totalArea")
            kitchenArea = form.cleaned_data.get("kitchenArea")
            latitude = form.cleaned_data.get("latitude")
            longitude = form.cleaned_data.get("longitude")
            filename = 'C:\\tmp\\finalized_model.sav'
             # load the model from disk
            loaded_model = pickle.load(open(filename, 'rb'))
            # прогноз для Самойловой. Вот ее квартира:
            X_test = np.array([[floorNumber, floorsTotal, totalArea, kitchenArea,latitude,longitude,]])  #55.786698
            # Предскажем ей цену
            predicted = loaded_model.predict(X_test)  # предсказываем цену квартиры Самойловой
            msg = 'Предсказала отлично!'
            return render(request, 'ML2.html', {'form': form, 'message': msg, 'predicted': predicted, })
    else:
        form = User_Haus_Form()
        return render(request, 'ML2.html', {'form': form, 'message': msg, })


from django.shortcuts import render
import pickle
import numpy as np

def index(request):
    return render(request, "index.html")

from .forms import User_bank_Form


def ML3(request):
    msg ='Заполните поля'
    if request.method == "POST":
       form = User_bank_Form(request.POST)
       if form.is_valid():
           kod_city = form.cleaned_data.get("kod_city")
           age = form.cleaned_data.get("age")
           money = form.cleaned_data.get("money")
           filename = 'C:\\tmp\\finalized_modelNB.sav'  #
           # load the model from disk
           loaded_model = pickle.load(open(filename, 'rb'))
           # прогноз для Самойловой. Она в этом банке
           x_test = np.array([[kod_city, age, money]])  # это Самойлова, она в этом банке  -Париж, возраст, вклад
           predicted = loaded_model.predict(x_test)
           # Предсказали ей уход из банка
           msg = 'Предсказала отлично!'
           return render(request, 'ML3.html', {'form': form, 'message': msg, 'predicted': predicted,})
    else:
        form = User_bank_Form()
        return render(request, 'ML3.html', {'form': form, 'message': msg,})


#Установить Keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
def ML1(request):
 if request.method == 'POST':
    IM_name = request.POST.get('radio')
    # Создаем модель с архитектурой VGG16 и загружаем веса, обученные
    # на наборе данных ImageNet
    model = VGG16(weights='imagenet')
    # Загружаем изображение для распознавания, преобразовываем его в массив
    # numpy и выполняем предварительную обработку
    img_path = 'C:\\tmp\\TF\\' + IM_name + '.jpg'  # 'ship.jpg' 'cat.jpg' 'plane.jpgэ cat1.jpg
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Запускаем распознавание объекта на изображении
    preds = model.predict(x)
    # Печатаем три класса объекта с самой высокой вероятностью
    #print('Результаты распознавания от Самойловой:', decode_predictions(preds, top=3)[0])
    # Печать наиболее вероятного
    from tensorflow.keras.applications.vgg16 import decode_predictions
    # convert the probabilities to class labels
    label = decode_predictions(preds)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    #print('%s (%.2f%%)' % (label[1], label[2] * 100))
    poroda = label[1]
    ver = label[2]
    return render(request, "ML1.html", {"IM_name": IM_name, "por": poroda, "ver": ver,})
 else:
       IM_name = 'first'
       return render(request, "ML1.html", {"IM_name": IM_name,"por": "нет", "ver": "нет", })


