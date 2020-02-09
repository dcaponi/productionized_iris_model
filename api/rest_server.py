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

    PG_HOST = 'database'
    PG_PORT = '5432'
    PG_USER = 'user'
    PG_PASS = 'password'
    PG_DB   = 'development'

    con_str = ('postgresql://{username}:{password}@{host}:{port}/{dbname}'.format(
        username=PG_USER,
        password=PG_PASS,
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB
    ))
    cnx = create_engine(con_str)

    specimen = np.append(specimen, pred_value)
    pd.DataFrame(specimen.reshape(-1, len(specimen)), columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'species']).to_sql('iris', cnx, if_exists='append')

    return {'prediction': prediction}

@app.route('/feedback', methods=['POST'])
def feedback():
    """do stuff"""

@app.route('/train', methods=['POST'])
def train():
    PG_HOST = 'database'
    PG_PORT = '5432'
    PG_USER = 'user'
    PG_PASS = 'password'
    PG_DB   = 'development'

    con_str = ('postgresql://{username}:{password}@{host}:{port}/{dbname}'.format(
        username=PG_USER,
        password=PG_PASS,
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB
    ))

    cnx = create_engine(con_str)
    data = pd.read_sql('iris', cnx)[['sepal length', 'sepal width', 'petal length', 'petal width', 'species']]
    X = data['sepal length', 'sepal width', 'petal length', 'petal width']
    Y = data['species']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    random_forest = RandomForestClassifier(
        criterion = "gini",
        min_samples_leaf = 3,
        min_samples_split = 2,
        max_depth=100,
        n_estimators=1000,
        max_features='auto',
        random_state=1,
        n_jobs=-1
    )
    random_forest.fit(X_train, Y_train)
    from sklearn.externals import joblib
    filename = 'iris_random_forest.sav'
    joblib.dump(random_forest, filename)

    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    precision = precision_score(Y_test, rf_predictions, average='micro')
    recall = recall_score(Y_test, rf_predictions, average='micro')

    return {'precision': precision, 'recall': recall}

if __name__ == '__main__':
    app.run(host='0.0.0.0')
