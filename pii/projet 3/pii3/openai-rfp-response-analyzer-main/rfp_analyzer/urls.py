from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('rfp_analyzer.analyzer.urls')),
    path('api/process-documents/', views.process_documents, name='process_documents'),
    path('api/regenerate-text/', views.regenerate_text, name='regenerate_text'),
    path('api/chat-message/', views.chat_message, name='chat_message'),
    path('api/generate-report/', views.generate_report, name='generate_report'),
    path('chat/', views.chat_view, name='chat_view'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 