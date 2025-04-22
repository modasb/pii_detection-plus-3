from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.AIRequestViewSet.as_view({'post': 'create'}), name='pii-detect'),
] 