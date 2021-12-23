from dataclasses import dataclass


@dataclass
class User:
    """
    classe que define um usuário
    Args:
        user_id: id do usuário
        username: nome de usuário
        first_name: primeiro nome do usuário
        last_name: sobrenome do usuário
    """
    user_id: int
    username: str
    first_name: str
    last_name: str
