import spark_prediction as sp
import pymysql
import pymysql.cursors
import os
import datetime
import time

userName = os.environ["MYSQL_USER"]
password = os.environ["MYSQL_PASSWORD"]
host = os.environ["MYSQL_HOST"]
port = 3306
dbname = "stocktwits"
        
conn = pymysql.connect(host, user=userName, port=port, password=password, db=dbname)
conn_cur = conn.cursor()

create_sql = "CREATE TABLE IF NOT EXISTS sentiment (tweet_id INT AUTO_INCREMENT PRIMARY KEY, " \
                       "createtime DATETIME(0), symbol VARCHAR(255), " \
                       "score FLOAT, prediction INT)";

conn_cur.execute(create_sql)
print("Test table created.")

stocks = sp.loc[["CreateTime", "Symbol", "score", "prediction"]]

start_time=datetime.datetime.now()

sqlInsert = """
            INSERT INTO sentiment(createtime, symbol, score, prediction) VALUES(%s,%s,%s,%s);
            """

f = '%Y-%m-%d %H:%M:%S'
for index,row in stocks.iterrows():
    temp = datetime.datetime.strptime(str(row[0]),f)
    conn_cur.execute(sqlInsert,(temp, row[1],row[2],row[3]))

conn.commit()
conn.close()

end_time=datetime.datetime.now()

print("Start time is {0}, end time is {1}".format(start_time, end_time))

