from django.contrib import admin
from django.urls import path, include
from pii_protection.views import home

urlpatterns = [
    path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('api/', include('pii_protection.urls')),
] 