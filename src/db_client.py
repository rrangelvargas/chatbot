import psycopg2
from .config import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST


# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]
#
#
# class DBClient(_DBClient, metaclass=Singleton):
#     pass


class DBClient:
    def __init__(self, dbname, user, password, host):
        self.connection = psycopg2.connect(f"dbname={dbname} user={user} password={password} host={host}")

    def execute_query(self, query, values=None):
        cur = self.connection.cursor()

        cur.execute(query, values)
        result = cur.fetchall()

        cur.close()
        # commit the changes
        self.connection.commit()

        return result


PostgresClient = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
