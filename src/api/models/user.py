from dataclasses import dataclass


@dataclass
class User:

    user_id: int
    username: str
    first_name: str
    last_name: str
