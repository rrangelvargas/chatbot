from datetime import datetime
from dataclasses import dataclass


@dataclass
class Message:
    """
    classe que define uma mensagem entre o usuário e o bot
    Args:
        id: id da mensagem
        text: conteúdo da mensagem
        user_id: id do usuário que enviou a mensagem
        sent_at: data e hora de envio da mensagem
    """
    id: int
    text: str
    user_id: int
    sent_at: datetime
