from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('process-documents/', views.process_documents, name='process_documents'),
    path('generate-report/', views.generate_report, name='generate_report'),
    path('chat/', views.chat_view, name='chat_view'),
    path('api/chat/', views.chat, name='chat_api'),
] 