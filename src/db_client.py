import psycopg2
from src.config import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST


class DBClient:
    """
    classe que define uma instância do cliente do pPostgresql
    """
    def __init__(self, dbname, user, password, host):
        """
        método de inicialização do cliente
        Args:
            dbname: nome do banco de dados a ser conectado
            user: usuário para acessar o banco de dados
            password: senha para acessar o banco de dados
            host: endereço do banco de dados
        """
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.connection = None

        self.connect()
        self.start_db()

    def connect(self):
        """
        método que inicia a conexão com o banco de dados
        """
        cn_str = f"dbname={self.dbname} user={self.user} password={self.password} host={self.host}"
        self.connection = psycopg2.connect(cn_str)

    def start_db(self):
        """
        método que cria as tabelas no banco de dados
        """

        # query para criar as tabelas
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
        """
        método de execução das queries pelo psycopg2
        Args:
            query: string contento a query desejada
            values: valores das variáveis da query

        Returns: resultado da query

        """

        # obtendo o cursor da conexão
        cur = self.connection.cursor()

        # executando a query e obtendo o resultado
        cur.execute(query, values)
        result = None
        if cur.description:
            result = cur.fetchall()

        # fechando o cursor da conexão e salvando os resultados
        cur.close()
        self.connection.commit()

        return result


# instância global do cliente Psycopg2
PostgresClient = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
