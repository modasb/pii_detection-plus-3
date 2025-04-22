import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Django settings
SECRET_KEY = 'your-secret-key'
DEBUG = True

# Database settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Basic Django settings
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'pii_protection',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'pii_protection.middleware.PIIProtectionMiddleware',
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

ROOT_URLCONF = 'pii_project.urls'

# Your existing PII and API Gateway settings...

# API Gateway settings
API_GATEWAY = {
    'ENDPOINTS': {
        'openai': 'https://api.openai.com/v1/chat/completions',
        # Add other AI service endpoints as needed
    },
    'TIMEOUT': 30,  # seconds
    'RETRY_ATTEMPTS': 3
}

# PII Protection Layer Settings
PII_PROTECTION_SETTINGS = {
    'ENGINE': 'presidio',
    'ENTITIES': [
        'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
        'CREDIT_CARD', 'IP_ADDRESS', 'US_SSN',
        'LOCATION', 'DATE_TIME'
    ],
    'CACHE_ENABLED': True,
    'CACHE_TIMEOUT': 3600,  # 1 hour
    'LOG_LEVEL': 'INFO'
}

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static'
]

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# WSGI application
WSGI_APPLICATION = 'pii_project.wsgi.application'

# Add to your settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'pii_protection': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}