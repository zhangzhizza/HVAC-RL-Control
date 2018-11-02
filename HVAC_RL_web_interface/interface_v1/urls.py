from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_all_exp/<slug:run_name>/', views.get_all_exp, name='get_all_exp'),
    re_path(r'^get_worker_status/$', views.get_worker_status, name='get_worker_status'),
    path('get_exp_status/<slug:run_name>/', views.get_exp_status, name='get_exps_status'),
    re_path(r'^run_exp/$', views.run_exp, name='run_exp'),
]