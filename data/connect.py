# https://www.geeksforgeeks.org/python-sqlite-connecting-to-database/
import sqlite3

try:
    conn = sqlite3.connect("D:\General Index\slice0\doc_keywords_0.sql\doc_keywords\doc_keywords_0.sql")
    cursor = conn.cursor()
    print("initialized SQlite DB")

    query = "select * from dkey, keywords_lc"
    cursor.execute(query)

    result = cursor.fetchall()
    print("Result:{}".format(result))

    cursor.close()

except sqlite3.Error as error:
    print("Error: ", error)

finally:

    if (conn):
        conn.close()
        print("Connection closed")