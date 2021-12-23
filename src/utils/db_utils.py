from src import PostgresClient


def count_sessions():
    """
    método para obter o número de sessões
    Returns: número de sessões

    """
    return PostgresClient.execute_query("SELECT COUNT(id) FROM session")[0][0]


def count_conversations():
    """
    método para obter o número de conversas
    Returns: número de conversas

    """
    return PostgresClient.execute_query("SELECT COUNT(id) FROM conversation")[0][0]


def count_messages():
    """
    método para obter o número de mensagens
    Returns: número de mensagens

    """
    return PostgresClient.execute_query("SELECT COUNT(id) FROM message")[0][0]
