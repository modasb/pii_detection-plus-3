from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyzer/process-files/', views.process_files, name='process_files'),
    path('analyzer/generate-analysis/', views.generate_analysis, name='generate_analysis'),
    path('analyzer/chat/', views.chat, name='chat'),
] 