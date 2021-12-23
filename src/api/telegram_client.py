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
    """
    classe que define um cliente do telegram
    Args:
        updater: objeto Updater
        dispatcher: objeto dispatcher
        sessions: diconário contendo todas as sessões ativas
        ml_client: instância do modelo de rede neural
        bot_id: id do bot no Telegram
        username: noem de usuário do bot no Telegram
        first_name: primeiro nome do bot
        last_name: sobrenome do bot
    """
    updater: Updater
    dispatcher: Dispatcher
    sessions: T.Dict[str, Session]
    ml_client: Model
    bot_id: int
    username: str
    first_name: str
    last_name: str

    def __init__(self, token: str):
        '''
        método de instanciação do cleinte do Telegram
        Args:
            token: token do bot para poder criar a conexão
        '''

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        # criação do bot
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.create_bot()

        # obtendo as sessões do banco de dados
        self.sessions = {}
        self.collect_sessions()

        # criando os comando básicos do bot
        start_handler = CommandHandler('start', self.start)
        training_handler = CommandHandler('train', self.train)
        message_handler = MessageHandler(Filters.text & (~Filters.command), self.send_message)

        self.dispatcher.add_handler(start_handler)
        self.dispatcher.add_handler(training_handler)
        self.dispatcher.add_handler(message_handler)

        # instanciando o modelo de rede neural
        self.ml_client = Model()
        self.ml_client.run('data/input/result.csv', 'data/output/pt_model/2-2_500/1000_checkpoint.tar')

        # self.ml_client.run('data/input/pairs.csv', 'data/output/cb_model/2-2_500/4000_checkpoint.tar')
        # self.ml_client.run('data/input/result.csv')
        # self.ml_client.train()

    def create_bot(self):
        """
        método para criar o bot no banco de dados
        """
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
        """
        método para obter as sessões do banco de dados e construir o dicionário de sessões
        """

        result = PostgresClient.execute_query(
            f'''
            SELECT id, data
            FROM session;
            '''
        )

        # para cada sessão é chamado o método de deserialização
        for session in result:
            self.sessions[session[0]] = deserialize_session(session[1])

    def start(self, update, context):
        """
        método para responder o comando \start quando o usuário fala com o bot pela primeira vez
        Args:
            update: última atualização da conversa
            context: contexto da conversa
        """

        answer = "Eu sou um robô, fale comigo!"
        # answer = "I'm a bot, please talk to me!"

        self.handle_message(update, answer)
        context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    def train(self, update, context):
        '''
        método para responder o comando \train e treinar o bot novamente
        Args:
            update: última atualização da conversa
            context: contexto da conversa
        '''
        pass

    def send_message(self, update, context):
        """
        método para obter a resposta do modelo de rede neural e responder ao usuário
        Args:
            update: última atualização da conversa
            context: contexto da conversa
        """

        try:
            # obtendo resposta da rede neural
            answer = format_answer(self.ml_client.evaluate(update.message.text))
        except KeyError:
            #caso a rede neural não consiga responder
            answer = "Desculpe, não consegui entender"
            # answer = "Sorry, i couldn't understand."

        # método para salvar a nova mensagem no abnco de dados
        self.handle_message(update, answer)
        context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    def handle_message(self, update, answer):
        """
        método para salvar a nova conversa no banco de dados e atualizar a conversa e a sessão
        Args:
            update: última atualização da conversa
            answer: mensagem à ser armazenada
        """
        user_id = update.message.chat_id

        # verificando se já existe uma sessão para a quele usuário
        if user_id not in self.sessions:
            user_info = update.message.chat

            # criando uma nova conversa
            new_conversation = Conversation(count_conversations()+1)

            # criando um novo usuário
            new_user = User(
                user_id,
                user_info.username,
                user_info.first_name,
                user_info.last_name
            )

            # criando uma nova sessão
            new_session = Session(user_id, new_conversation, new_user, None)

            # criando uma nvoa mensagem
            new_session.new_message(update.message.text, user_id, datetime.now())

            self.sessions[user_id] = new_session

        else:
            # criando uma nova mensagem
            self.sessions[user_id].new_message(update.message.text, user_id, datetime.now())

        self.sessions[user_id].new_message(answer, self.bot_id, datetime.now())

    def run(self):
        '''
        método para rodar o cliente do Telegram
        '''
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
