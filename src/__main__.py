from api import Client
from config import TELEGRAM_BOT_TOKEN

if __name__ == "__main__":
    client = Client(TELEGRAM_BOT_TOKEN)
    client.run()
