from datetime import datetime
from dataclasses import dataclass
import typing as T
from .message import Message
from src import PostgresClient


@dataclass
class Conversation:
    id: int
    messages: T.List[Message]
    started_at: datetime

    def __init__(self, conversation_id: int):
        self.id = conversation_id
        self.started_at = datetime.now()
        self.messages = []

    def add_message(self, message: Message, session_id: int):
        self.messages.append(message)

        add_message_entry = f'''
            INSERT INTO message(id, session_id, sent_at, text)
            VALUES {message.id, session_id, message.sent_at, message.text};
        '''
        add_conversation_message_entry = f'''
            INSERT INTO conversation_message(message_index, conversation_id, message_id)
            VALUES {len(self.messages), self.id, message.id};
        '''

        PostgresClient.execute_query(add_message_entry)
        PostgresClient.execute_query(add_conversation_message_entry)
