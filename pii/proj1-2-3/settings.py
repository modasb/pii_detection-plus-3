import os

# Django settings
SECRET_KEY = 'your-secret-key'
DEBUG = True

# PII Protection Layer Settings
PII_PROTECTION_SETTINGS = {
    'ENGINE': 'presidio',  # or 'local_llm'
    'ENTITIES': [
        'PERSON',
        'EMAIL_ADDRESS',
        'PHONE_NUMBER',
        'CREDIT_CARD',
        'IP_ADDRESS',
        'US_SSN',
        'LOCATION',
        'DATE_TIME'
    ],
    'CACHE_ENABLED': True,
    'CACHE_TIMEOUT': 3600,  # 1 hour
    'LOG_LEVEL': 'INFO'
}

# API Gateway settings
API_GATEWAY = {
    'ENDPOINTS': {
        'openai': 'https://api.openai.com/v1/chat/completions',
        # Add other AI service endpoints as needed
    },
    'TIMEOUT': 30,  # seconds
    'RETRY_ATTEMPTS': 3
}

# Database settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db.sqlite3',
    }
}

# Add 'rest_framework' to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'your_app_name',  # Replace with your actual app name
]

# Add the PII middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'your_app_name.middleware.PIIProtectionMiddleware',  # Add this line
] 