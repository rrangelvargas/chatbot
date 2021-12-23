from datetime import datetime
from dataclasses import dataclass, field
import typing as T
from .message import Message
from src import PostgresClient
from src.utils import datetime_to_timestamp


@dataclass
class Conversation:
    """
    classe que define uma conversa entre o usuário e o bot
    Args:
        id: id da conversa
        messages: lista de mensagens da conversa
        started_at: data e hora de inicio da conversa
    """
    id: int
    messages: T.List[Message] = field(default_factory=list)
    started_at: datetime = datetime.now()

    def add_message(self, message: Message, session_id: int):
        """
        método que adiciona uma mensagem à conversa
        Args:
            message: mensagem a ser adicoonada
            session_id: id da sessão associada à conversa
        """

        # adiciona a mensagem na lista de mensagens
        self.messages.append(message)

        # query para salvar a mensagem no banco de dados
        add_message_entry = f'''
            INSERT INTO message(id, session_id, user_id, sent_at, text)
            VALUES {
                message.id,
                session_id,
                message.user_id,
                datetime_to_timestamp(message.sent_at),
                message.text
            }
            ON CONFLICT DO NOTHING
            RETURNING id;
        '''

        # query para salvar a relação conversa-mensagem no banco de dados
        add_conversation_message_entry = f'''
            INSERT INTO conversation_message(message_index, conversation_id, message_id)
            VALUES {len(self.messages), self.id, message.id}
            ON CONFLICT DO NOTHING
            RETURNING id;
        '''

        PostgresClient.execute_query(add_message_entry)
        PostgresClient.execute_query(add_conversation_message_entry)

