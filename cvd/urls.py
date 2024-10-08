from django.urls import path
from . import views

urlpatterns = [
    path('cvd/', views.predict, name='cvd'),
]
