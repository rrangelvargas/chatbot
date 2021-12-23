import json
from dataclasses import dataclass, asdict
from datetime import datetime
from .message import Message
from .conversation import Conversation
from .user import User
from src import PostgresClient
from src.utils import count_messages, count_conversations, datetime_to_timestamp
import typing as T
from src.utils import normalize_string


@dataclass
class Session:
    """
    classe que define uma sessão entre o usuário e o bot
    Args:
        id: id da sessão
        current_conversation: atual conversa da sessão
        user: usuário associado à sessão
        last_message: última mensagem enviada na sessão
        correct_answer: flag para determinar se o bot está recebendo a resposta correta para uma mensagem
    """
    id: int
    current_conversation: Conversation
    user: User
    last_message: T.Optional[Message]
    correct_answer: bool

    def __init__(
            self,
            session_id: int,
            current_conversation: Conversation,
            user: User,
            last_message: T.Optional[Message]
    ):
        """
        método para inicializar uma nova sessão
        Args:
            session_id: id da nova sessão
            current_conversation: atual conversa da nova sessão
            user: usuário associado à sessão
            last_message: última mensagem enviada na sessão
        """
        self.id = session_id
        self.current_conversation = current_conversation
        self.user = user
        self.last_message = last_message
        self.correct_answer = False

        # criando a sessão no banco de dados
        self.create_new_session()

    def new_message(self, message: str, user_id: int, sent_at: datetime):
        new_message = Message(count_messages()+1, normalize_string(message), user_id, sent_at)

        if self.last_message and new_message.sent_at.hour - self.last_message.sent_at.hour > 1:
            update_conversation = f'''
                UPDATE conversation
                SET ended_at = '{datetime_to_timestamp(self.last_message.sent_at)}'
                WHERE id = {self.current_conversation.id}
                RETURNING id;
            '''
            PostgresClient.execute_query(update_conversation)

            new_conversation = Conversation(count_conversations()+1)

            create_new_conversation = f'''
                INSERT INTO conversation(session_id, started_at, id)
                VALUES {self.id, datetime_to_timestamp(new_conversation.started_at), new_conversation.id}
                ON CONFLICT DO NOTHING
                RETURNING id;            
            '''

            PostgresClient.execute_query(create_new_conversation)

            self.current_conversation = new_conversation

            update_session = f'''
                UPDATE session
                SET current_conversation_id = {self.current_conversation.id},
                    data = '{json.dumps(asdict(self), default=str)}'
                WHERE id = {self.id}
                RETURNING id;
            '''
            PostgresClient.execute_query(update_session)

        self.current_conversation.add_message(new_message, self.id)
        self.last_message = new_message

        update_session_last_message = f'''
            UPDATE session
            SET last_message_id = {new_message.id},
                data = '{json.dumps(asdict(self), default=str)}'
            WHERE id = {self.id}
            RETURNING id;
        '''
        PostgresClient.execute_query(update_session_last_message)

    def create_new_session(self):
        """
        método para criar a sessão no banco de dados, o usuário associado e a nova conversa
        """

        # query para criar uma nova sessão
        new_session = f'''
            INSERT INTO session(id, user_id, current_conversation_id, data)
            VALUES {
                self.id,
                self.user.user_id,
                self.current_conversation.id,
                json.dumps(asdict(self), default=str)
            }
            ON CONFLICT DO NOTHING
            RETURNING id;
        '''

        PostgresClient.execute_query(new_session)

        # query para criar um novo usuário
        new_user = f'''
            INSERT INTO telegram_user(id, session_id, username, first_name, last_name)
            VALUES {self.user.user_id, self.id, self.user.username, self.user.first_name, self.user.last_name}
            ON CONFLICT DO NOTHING
            RETURNING id;        
        '''

        PostgresClient.execute_query(new_user)


        # query para criar a conversa
        new_conversation = f'''
            INSERT INTO conversation(id, session_id, started_at)
            VALUES {
                self.current_conversation.id,
                self.id,
                datetime_to_timestamp(self.current_conversation.started_at)
            }
            ON CONFLICT DO NOTHING
            RETURNING id;
        '''

        PostgresClient.execute_query(new_conversation)
