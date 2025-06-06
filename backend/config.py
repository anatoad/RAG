import os

class Config:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    DEBUG = True
    CORS_HEADERS = 'Content-Type'
