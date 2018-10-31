from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_all_exp/<slug:run_name>/', views.get_all_exp, name='get_all_exp')
]