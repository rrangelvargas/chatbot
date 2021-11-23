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
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.connection = None

        self.connect()
        self.start_db()

    def connect(self):
        cn_str = f"dbname={self.dbname} user={self.user} password={self.password} host={self.host}"
        self.connection = psycopg2.connect(cn_str)

    def start_db(self):
        query = f'''
            create table if not exists session
            (
                id                      integer not null
                    constraint session_pkey
                        primary key,
                user_id                 integer not null,
                current_conversation_id integer not null,
                last_message_id         integer,
                last_message_date       timestamp,
                data                    json
            );
            
            alter table session
                owner to {self.user};
            
            create table if not exists telegram_user
            (
                id         integer not null
                    constraint telegram_user_pkey
                        primary key,
                session_id integer,
                username   text,
                first_name text,
                last_name  text
            );
            
            alter table telegram_user
                owner to {self.user};
            
            create table if not exists conversation
            (
                id         integer not null
                    constraint conversation_pkey
                        primary key,
                session_id integer,
                started_at timestamp,
                ended_at   timestamp
            );
            
            alter table conversation
                owner to {self.user};
            
            create table if not exists message
            (
                user_id    integer not null,
                id         integer not null
                    constraint message_pkey
                        primary key,
                session_id integer,
                text       text,
                sent_at    timestamp
            );
            
            alter table message
                owner to {self.user};
            
            create table if not exists conversation_message
            (
                id              serial
                    constraint conversation_message_pkey
                        primary key,
                conversation_id integer
                    constraint conversation_message_conversation_id_fkey
                        references conversation,
                message_id      integer
                    constraint conversation_message_message_id_fkey
                        references message,
                message_index   integer
            );
            
            alter table conversation_message
                owner to {self.user};
        '''
        self.execute_query(query)

    def execute_query(self, query, values=None):

        cur = self.connection.cursor()

        cur.execute(query, values)
        print(cur.description)
        result = None
        if cur.description:
            result = cur.fetchall()

        cur.close()
        # commit the changes
        self.connection.commit()

        return result


PostgresClient = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
