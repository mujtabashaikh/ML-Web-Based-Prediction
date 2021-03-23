from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

model = pickle.load(open('logistic.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    if request.method=='POST':
        sepal_length=request.form['s_length']
        sepal_width=request.form['s_width']
        petal_length=request.form['p_length']
        petal_width=request.form['p_width']
        data=[[float(sepal_length),float(sepal_width),float(petal_length),float(petal_width)]]
        model=joblib.load('logistic.pkl')
        predict_result=model.predict(data)
        

    return render_template('result.html',sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,predict_result=predict_result)


# @app.route('/predict', methods=['POST'])
# def home():
#     data1 = request.form['a']
#     data2 = request.form['b']
#     data3 = request.form['c']
#     data4 = request.form['d']
#     ar = np.array([[data1, data2, data3, data4]])
#     pred = model.predict(ar)
#     return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















