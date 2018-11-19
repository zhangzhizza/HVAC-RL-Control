from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test/', views.test),
    re_path(r'^simulator_eplus/openjscad/$', views.openJSCAD, name='openJSCAD'),
    re_path(r'^simulator_eplus/get_aval_schedules/$', views.generate_idf_schedule_names, name='generate_idf_schedule_names'),
    path('simulator_eplus', views.simulator_eplus, name='simulator_eplus'),
    path('simulator_eplus/upload_idf', views.simulator_eplus_idf_upload, name='simulator_eplus_idf_upload'),
    path('get_all_exp/<slug:run_name>/', views.get_all_exp, name='get_all_exp'),
    re_path(r'^get_worker_status/$', views.get_worker_status, name='get_worker_status'),
    path('get_exp_status/<slug:run_name>/', views.get_exp_status, name='get_exps_status'),
    re_path(r'^run_exp/$', views.run_exp, name='run_exp'),
    re_path(r'^get_eval_res_hist/$', views.get_eval_res_hist, name='get_eval_res_hist'),   
]