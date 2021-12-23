from src.api.models import User, Conversation, Message, Session
from datetime import datetime


def deserialize_session(session_dict):
    """
    método que recebe um dicionário com dados de uma sessão e instaância um objeto de Session com esses dados
    Args:
        session_dict: diconário contendo dados de uma sessão

    Returns: objeto de Session

    """
    messages = []

    # construção dos objetos Message referentes às mensagens da conversa atual
    for message in session_dict['current_conversation']['messages']:
        messages.append(
            Message(
                message['id'],
                message['text'],
                message['user_id'],
                datetime.strptime(message['sent_at'], '%Y-%m-%d %H:%M:%S.%f')))

    # construção do objeto Conversation referente à conversa atual
    current_conversation = Conversation(session_dict['current_conversation']['id'])
    current_conversation.started_at = datetime.strptime(
        session_dict['current_conversation']['started_at'], '%Y-%m-%d %H:%M:%S.%f'
    )
    current_conversation.messages = messages

    last_message = messages[-1]

    # Construção do objeto User referente ao usuário da sessão
    user = User(
        session_dict['user']['user_id'],
        session_dict['user']['username'],
        session_dict['user']['first_name'],
        session_dict['user']['last_name']
    )

    # contrução do objeto Session
    session = Session(
        session_id=session_dict['id'],
        current_conversation=current_conversation,
        last_message=last_message,
        user=user
    )

    return session
