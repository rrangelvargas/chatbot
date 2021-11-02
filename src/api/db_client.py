import psycopg2


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
