import psycopg2


class DBClient:
    def __init__(self, dbname, user, password, host):
        self.connection = psycopg2.connect(f"dbname={dbname} user={user} password={password} host={host}")

    def execute_query(self, query, values=None):
        cur = self.connection.cursor()

        # query = 'CREATE TABLE user_example(ID INT PRIMARY KEY NOT NULL, NAME TEXT NOT NULL, AGE INT NOT NULL);'
        # query2 = 'INSERT INTO user_example(id, name, age) VALUES (%s, %s, %s);'
        # values = (0, "rodrigo2", 25)
        cur.execute(query, values)

        result = cur.fetchall()

        cur.close()
        # commit the changes
        self.connection.commit()

        return result
