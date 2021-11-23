from dataclasses import dataclass


@dataclass
class User:

    user_id: int
    username: str
    first_name: str
    last_name: str

    def __init__(self, user_id: int, username: str, first_name: str, last_name: str):
        self.user_id = user_id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
