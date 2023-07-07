from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('model_rf.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        
        year = int(request.form.get('year'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        sqft_living = int(request.form.get('sqft_living'))
        sqft_lot = int(request.form.get('sqft_lot'))
        floors = int(request.form.get('floors'))
        waterfront = int(request.form.get('waterfront'))
        view = int(request.form.get('view'))
        condition = int(request.form.get('condition'))
        sqft_above = int(request.form.get('sqft_above'))
        sqft_basement= int(request.form.get('sqft_basement'))
        yr_built= int(request.form.get('yr_built'))
        yr_renovated =int(request.form.get('yr_renovated'))
        statezip = int(request.form.get('statezip'))

        features = np.array([year, bedrooms, bathrooms, sqft_living, sqft_lot, floors,waterfront,view,condition, sqft_above,sqft_basement,yr_built,yr_renovated,statezip]).reshape(1, -1)
        prediction = model.predict(features)
        

        return str(prediction[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)


