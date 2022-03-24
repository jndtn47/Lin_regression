from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('ml1', views.ML1, name='ml1'),
    path('ml2', views.ML2, name='ml2'),
    path('ml3', views.ML3, name='ml3'),
]
