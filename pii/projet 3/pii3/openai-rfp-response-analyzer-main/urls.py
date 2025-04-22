from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_documents, name='process_documents'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('chat/', views.chat, name='chat'),
] 