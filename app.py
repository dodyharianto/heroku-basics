from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

model_filename = 'logreg_iris.sav'
model = pickle.load(open(model_filename, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods = ['POST'])
def prediction():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    print(data1)
    arr = np.array([[data1, data2, data3, data4]])
    predictions = model.predict(arr)
    return render_template('prediction.html', data = predictions)

if __name__ == '__main__':
    app.run(debug = True)