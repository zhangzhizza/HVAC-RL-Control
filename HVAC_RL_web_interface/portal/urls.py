from django.urls import path, re_path
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import logout_then_login
from . import views

urlpatterns = [
	path('login/', auth_views.LoginView.as_view(template_name='portal/html/login.html'), name = 'login'),
	path('logout/', logout_then_login, name = 'logout_then_login'),
	]