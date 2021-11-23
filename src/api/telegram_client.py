from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher
import logging
from dataclasses import dataclass
from src.api.models import Session, Conversation, User, deserialize_session
from src.utils import count_conversations, PostgresClient, format_answer
import typing as T
from datetime import datetime
from src.ml_model import Model


@dataclass
class Client:
    updater: Updater
    dispatcher: Dispatcher
    sessions: T.Dict[str, Session]
    ml_client: Model
    bot_id: int
    username: str
    first_name: str
    last_name: str

    def __init__(self, token: str):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.create_bot()

        self.sessions = {}
        self.collect_sessions()

        start_handler = CommandHandler('start', self.start)
        message_handler = MessageHandler(Filters.text & (~Filters.command), self.send_message)

        self.dispatcher.add_handler(start_handler)
        self.dispatcher.add_handler(message_handler)

        self.ml_client = Model()
        self.ml_client.run('data/input/pairs.csv', 'data/output/cb_model/2-2_500/4000_checkpoint.tar')

    def create_bot(self):
        self.bot_id = self.updater.bot.id
        self.username = self.updater.bot.username
        self.first_name = self.updater.bot.first_name
        self.last_name = 'bot'

        PostgresClient.execute_query(
            f'''
            INSERT INTO telegram_user(id, username, first_name, last_name)
            VALUES {self.bot_id, self.username, self.first_name, self.last_name}
            ON CONFLICT DO NOTHING
            RETURNING id;
            '''
        )

    def collect_sessions(self):
        result = PostgresClient.execute_query(
            f'''
            SELECT id, data
            FROM session;
            '''
        )

        for session in result:
            self.sessions[session[0]] = deserialize_session(session[1])

    def start(self, update, context):
        answer = "I'm a bot, please talk to me!"

        self.handle_message(update, answer)
        context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    def send_message(self, update, context):
        try:
            answer = format_answer(self.ml_client.evaluate(update.message.text))
        except KeyError:
            answer = "Sorry, i couldn't understand."
        self.handle_message(update, answer)
        context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    def handle_message(self, update, answer):
        user_id = update.message.chat_id

        if user_id not in self.sessions:
            user_info = update.message.chat

            new_conversation = Conversation(count_conversations()+1)
            new_user = User(
                user_id,
                user_info.username,
                user_info.first_name,
                user_info.last_name
            )
            new_session = Session(user_id, new_conversation, new_user, None)
            new_session.new_message(update.message.text, user_id, datetime.now())

            self.sessions[user_id] = new_session

        else:
            self.sessions[user_id].new_message(update.message.text, user_id, datetime.now())

        self.sessions[user_id].new_message(answer, self.bot_id, datetime.now())

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