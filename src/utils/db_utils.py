from src import PostgresClient


def count_conversations():
    return PostgresClient.execute_query("SELECT COUNT(id) FROM conversation")


def count_messages():
    return PostgresClient.execute_query("SELECT COUNT(id) FROM message")
