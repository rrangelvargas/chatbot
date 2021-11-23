from datetime import datetime
from dataclasses import dataclass
from src.utils import normalize_string


@dataclass
class Message:

    id: int
    text: str
    user_id: int
    sent_at: datetime

    def __init__(self, message_id: int, text: str, user_id: int, sent_at: datetime):
        self.id = message_id
        self.text = normalize_string(text)
        self.user_id = user_id
        self.sent_at = sent_at
