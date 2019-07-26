from django.urls import path
from . import views

urlpatterns = [
    path('dq_bot/',views.dq_bot, name='dq_bot'),
]