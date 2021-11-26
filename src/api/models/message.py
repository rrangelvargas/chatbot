from datetime import datetime
from dataclasses import dataclass


@dataclass
class Message:

    id: int
    text: str
    user_id: int
    sent_at: datetime
