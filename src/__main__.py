from api import Client
from config import TELEGRAM_BOT_TOKEN

if __name__ == "__main__":
    # inicializando o bot do telegram
    client = Client(TELEGRAM_BOT_TOKEN)
    client.run()
