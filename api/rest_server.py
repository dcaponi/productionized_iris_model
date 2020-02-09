from flask import Flask
from flask_restful import Resource, Api, reqparse
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib
import numpy as np

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

@app.route('/prediction', methods=['POST'])
def predict():
    parser.add_argument('sepal_length', type=float)
    parser.add_argument('sepal_width', type=float)
    parser.add_argument('petal_length', type=float)
    parser.add_argument('petal_width', type=float)

    args = parser.parse_args()
    specimen = np.array( [ args['sepal_length'], args['sepal_width'], args['petal_length'], args['petal_width'] ] )

    choices = ['setosa', 'versicolor', 'virginica']
    loaded_model = joblib.load(open('random_forest', 'rb'))

    pred_value = loaded_model.predict(np.array([specimen]))
    prediction = choices[pred_value[0]]

    return {'prediction': prediction}

if __name__ == '__main__':
    app.run(host='0.0.0.0')
