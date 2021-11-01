import psycopg2

conn = psycopg2.connect("dbname=postgres user=postgres password=postgres")

# create a cursor
cur = conn.cursor()

# execute a statement
print('PostgreSQL database version:')
# query = 'CREATE TABLE user_example(ID INT PRIMARY KEY NOT NULL, NAME TEXT NOT NULL, AGE INT NOT NULL);'
# cur.execute(query)
query2 = 'INSERT INTO user_example(name, age) VALUES (%s, %s, %s);'
values = (0, "rodrigo", 24)
cur.execute(query2, values)

cur.close()
# commit the changes
conn.commit()

# # display the PostgreSQL database server version
# db_version = cur.fetchone()
# print(db_version)