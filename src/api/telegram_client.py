# 1347095513:AAExt4kUogFel6ZOlxVDvalbW4DcNvosvl8

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging

class Client:
    def __init__(self):
        self.updater = Updater(token='1347095513:AAExt4kUogFel6ZOlxVDvalbW4DcNvosvl8', use_context=True)
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        self.dispatcher = self.updater.dispatcher

        start_handler = CommandHandler('start', self.start)
        message_handler = MessageHandler(Filters.text & (~Filters.command), self.send_message)

        self.dispatcher.add_handler(start_handler)
        self.dispatcher.add_handler(message_handler)

    def start(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


    def send_message(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

    def run(self):
        try:
            while True:
                if not self.updater.running:
                    print("Starting bot...")
                    self.updater.start_polling()
                    print("Your bot is running!")
        except KeyboardInterrupt:
            print("Stopping bot...")
            self.updater.stop()
            print("Done!")


if __name__== "__main__":
  client = Client()
  client.run()



