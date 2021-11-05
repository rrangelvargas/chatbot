from dataclasses import dataclass


@dataclass
class User:

    id: int
    username: str
    first_name: str
    last_name: str

    def __init__(self, username: str, first_name: str, last_name: str):
        self.username = username
        self.first_name = first_name
        self.last_name = last_name

