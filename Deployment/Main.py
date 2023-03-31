from flask import Flask,request,render_template,redirect,url_for

import pymysql as sql
import pickle
from csv import writer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


app = Flask(__name__)

@app.route('/')
def func():
    return render_template("login.html")


@app.route('/checker',methods=["GET","POST"])
def checker():
    conn = sql.connect(host='localhost',user='root',password='Darkn@e1',db = 'EMP')
    cur = conn.cursor()
    cur.execute("select * from Employee")
    out = cur.fetchall()
    id = [i[1] for i in out ]
    passw = [i[2] for i in out]

    if(request.method=='POST'):

        data_uid = request.form['UID']
        if(data_uid in id and request.form['Pass']== passw[id.index(data_uid)]):
            return render_template("Predict.html",ITEM = "Hello " + data_uid)
        else:
            return render_template('login.html',data="*UserName Doesn't exist / Password Entered is Wrong.")



@app.route('/find',methods=['GET','POST'])
def find():
    plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})
    model = pickle.load(open('model.pkl','rb'))
    model_poly = pickle.load(open('model_poly.pkl','rb'))
    x_inp = model_poly.fit_transform(np.array(float(request.form['Exp'])).reshape(-1,1))
    out = model.predict(x_inp)
    d = pd.read_csv('model/Salary_Data.csv')
    print(out[0][0])
    s = pd.DataFrame({'YearsExperience':[float(request.form['Exp'])],'Salary':[out[0][0]]})
    # d.loc[len(d)] = [float(request.form['Exp']),out[0][0]]
    d.sort_values('YearsExperience',inplace=True,ascending=True)
    x = d['YearsExperience'].values
    y = d['Salary'].values
    d.to_csv('/home/vignesh/ML/Deployment/model/Salary_Data.csv')
    s = np.linspace(1,math.floor(float(request.form['Exp'])+4))
    plt.ylabel('Salary')
    plt.xlabel('Years Of Experience')
    plt.plot(x,y)
    plt.plot(s,model.predict(model_poly.fit_transform(s.reshape(-1,1))))
    plt.savefig('/home/vignesh/ML/Deployment/static/graph.png')
    return render_template("Predict.html",output = "The Salary you will be receieving is : {}".format(round(out[0][0],2)))

if(__name__=='__main__'):
    prev = None
    app.run(host='localhost',debug=True)