from src import PostgresClient


def count_sessions():
    return PostgresClient.execute_query("SELECT COUNT(id) FROM session")[0][0]


def count_conversations():
    return PostgresClient.execute_query("SELECT COUNT(id) FROM conversation")[0][0]


def count_messages():
    return PostgresClient.execute_query("SELECT COUNT(id) FROM message")[0][0]
