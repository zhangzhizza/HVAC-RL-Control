from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test/', views.test),
    re_path(r'^simulator_eplus/openjscad/$', views.openJSCAD, name='openJSCAD'),
    re_path(r'^simulator_eplus/get_aval_schedules/$', views.generate_idf_schedule_names, name='generate_idf_schedule_names'),
    re_path(r'^simulator_eplus/generate_minmax_limits/$', views.generate_minmax_limits, name='generate_minmax_limits'),
    re_path(r'^simulator_eplus/get_state_names/$', views.get_state_names, name='get_state_names'),
    re_path(r'^simulator_eplus/get_fileschedules/$', views.generate_idf_fileschedule_names, name='generate_idf_fileschedule_names'),
    re_path(r'^simulator_eplus/get_weathers/$', views.generate_epw_names, name='generate_epw_names'),
    re_path(r'^simulator_eplus/create_env/$', views.create_env, name='create_env'),
    path('simulator_eplus/get_all_envs/', views.get_all_envs, name='get_all_envs'),
    path('simulator_eplus', views.simulator_eplus, name='simulator_eplus'),
    path('simulator_eplus/upload_idf', views.simulator_eplus_idf_upload, name='simulator_eplus_idf_upload'),
    re_path(r'get_all_exp/$', views.get_all_exp, name='get_all_exp'),
    re_path(r'^get_worker_status/$', views.get_worker_status, name='get_worker_status'),
    re_path(r'get_exp_status/$', views.get_exp_status, name='get_exps_status'),
    re_path(r'^run_exp/$', views.run_exp, name='run_exp'),
    re_path(r'^reset_exp/$', views.reset_exp, name='reset_exp'),
    re_path(r'^get_eval_res_hist/$', views.get_eval_res_hist, name='get_eval_res_hist'),   
]