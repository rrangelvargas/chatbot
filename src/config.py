from decouple import config

# variáveis globais de configuração
TELEGRAM_BOT_TOKEN = config('TELEGRAM_BOT_TOKEN')
POSTGRES_DB = config('POSTGRES_DB')
POSTGRES_USER = config('POSTGRES_USER')
POSTGRES_PASSWORD = config('POSTGRES_PASSWORD')
POSTGRES_HOST = config('POSTGRES_HOST')
