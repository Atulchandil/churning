
#this si from clon repo in local
from flask import Flask,render_template,request
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler=MinMaxScaler()
df=pd.read_csv("for_scaler.csv")
df=df.iloc[:,1:]
scaler.fit(df)

app=Flask(__name__)

import numpy as np
import keras
from keras.models import load_model

def modelload():
    mo = load_model("C:\\Users\hriti\model.h5")
    print("model loaded")
    return mo
model=modelload()
print (type(model))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/submit",methods=["POST"])
def prediction():
    #HTML->python
    if request.method=="POST":
        lis=[]
        geography=request.form["geography"]
        customerid = request.form["customerid"]
        surname = request.form["surname"]
        creditscore = request.form["creditscore"]
        gender = request.form["gender"]
        age = request.form["age"]
        tenure = request.form["tenure"]
        balance = request.form["balance"]
        numofproducts = request.form["numofproducts"]
        hascrcard = request.form["hascrcard"]
        isactivemember = request.form["isactivemember"]
        estimatedsalary = request.form["estimatedsalary"]
        sentiment = request.form["sentiment"]
        lis.append(int(creditscore))
        if (geography == "France"):
            lis.append(0)
        if (geography == "Spain"):
            lis.append(2)
        if (geography == "Germany"):
            lis.append(1)
        lis.append(int(numofproducts))
        lis.append(int(isactivemember))
        lis.append(int(sentiment))
        if (int(age) >= 17 & int(age) < 32):
            lis.append(1)
        elif (int(age) >= 32 & int(age) < 40):
            lis.append(2)
        elif (int(age) >= 40 & int(age) < 50):
            lis.append(3)
        else:
            lis.append(4)
        arr = scaler.transform(np.array(lis).reshape(-1, 6))
        pred = np.argmax(model.predict(arr), axis=1)
        if (pred == 1):
            out = "customer may leave the bank"
        if (pred == 0):
            out = "customer may not leave the bank"
        return render_template("submit.html",score=out)


if __name__ == "__main__":
    app.run(debug=True)