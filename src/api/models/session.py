from dataclasses import dataclass
import typing as T
from .message import Message
from .conversation import Conversation
from .user import User
from src import PostgresClient
from src.utils import count_messages, count_conversations


@dataclass
class Session:

    id: int
    current_conversation: Conversation
    user: User
    last_message: Message

    def __init__(self, session_id: int, current_conversation: Conversation, user: User, last_message: Message):
        self.id = session_id
        self.current_conversation = current_conversation
        self.user = user
        self.last_message = last_message

        query = f'''
            INSERT INTO session(id, user_id, current_conversation_id, last_message_id, las_message_date)
            VALUES {
                self.id,
                self.user.id,
                self.current_conversation.id,
                self.last_message.id,
                self.last_message.sent_at
            };
        '''

        PostgresClient.execute_query(query)

    def new_message(self, message):
        new_message = Message(count_messages()+1, message)

        if new_message.sent_at.hour - self.last_message.sent_at.hour > 1:
            update_conversation = f'''
                UPDATE conversation
                SET ended_at = {self.last_message.sent_at}
                WHERE id = {self.current_conversation.id};
            '''
            PostgresClient.execute_query(update_conversation)

            new_conversation = Conversation(count_conversations()+1)

            create_new_conversation = f'''
                INSERT INTO conversation(session_id, ended_at, started_at, id)
                VALUES {self.id, None, new_conversation.started_at, new_conversation.id};            
            '''

            PostgresClient.execute_query(create_new_conversation)

            self.current_conversation = new_conversation

            update_session = f'''
                UPDATE session
                SET current_conversation_id = {self.current_conversation.id}
                WHERE id = {self.id};
            '''
            PostgresClient.execute_query(update_session)

        self.current_conversation.add_message(new_message, self.id)
        self.last_message = new_message

