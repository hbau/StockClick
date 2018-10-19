from flask import Flask, jsonify, render_template, request, redirect, url_for
import mysql.connector

app = Flask(__name__)

db_config={
	'host':'stocktwits.cggcdnio3ixi.us-east-1.rds.amazonaws.com',
	'user':'root',
	'password':'12345678',
	'database':'stocktwits'
}

connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

@app.route('/')
def query():
    print("I am in home ")
    return render_template("index.html")

@app.route('/score', methods=['POST'])
def score():
    symbol = str(request.form["symbol"])
    hour = int(request.form["hour"])
    minute = int(request.form["minute"])
    minute = int(minute)+int(hour)*60
    
    query = """
            SET time_zone = 'US/Eastern';
            """
    query2 = """
            SELECT ROUND(SUM(score),1) from sentiment2 
            WHERE (symbol = %s)
            AND (createtime > (SELECT DATE_SUB("2018-10-03 14:57:15", INTERVAL %s MINUTE)))
            """
    query3 = """
            SELECT COUNT(*) from sentiment2 
            WHERE (symbol = %s)
            AND (createtime > (SELECT DATE_SUB("2018-10-03 14:57:15", INTERVAL %s MINUTE)))
    """
        
    cursor.execute(query)
    valueData = (symbol,minute)
    cursor.execute(query2,valueData)
    score = cursor.fetchone()[0]
    cursor.execute(query3,valueData)
    number = cursor.fetchone()[0]
#    score = round(score/(number+1),2)
    result = [score, number]
    return render_template("score.html", data=result)

if __name__ == '__main__':
    app.run(debug=True)