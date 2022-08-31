from flask import Flask,request,jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
diabeticmodel=pickle.load(open('diabeticmodel.pkl','rb'))  #load the model
@app.route('/')
def home():
    return '<h1>Hello World!</h1>'

@app.route('/predict',methods=['POST'])
def predict():
    pregnancies=request.form.get('pregnancies')
    glucose=request.form.get('glucose')
    bloodpressure=request.form.get('bloodpressure')
    skinthickness=request.form.get('skinthickness')
    insulin=request.form.get('insulin')
    bmi=request.form.get('bmi')
    diabetespedigreefunction=request.form.get('diabetespedigreefunction')
    age=request.form.get('age')
    input_query=np.array([[pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]])
    result=diabeticmodel.predict(input_query)[0]
    return jsonify({"Diabetic":str(result)})


if __name__ == '__main__':
    app.run(debug=True)

