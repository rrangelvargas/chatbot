from datetime import datetime
from dataclasses import dataclass


@dataclass
class Message:

    id: int
    text: str
    sent_at: datetime

    def __init__(self, message_id: int, text: str):
        self.id = message_id
        self.text = text
        self.sent_at = datetime.now()