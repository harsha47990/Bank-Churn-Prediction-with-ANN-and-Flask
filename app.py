import keras
import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd

app = Flask(__name__,template_folder='template')
model =keras.models.load_model('.\Model\classifier')
ss = joblib.load('.\Model\StandardScalar')
main_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Germany', 'Spain',
       'Male']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features.pop()
    country = [0,0]
    if features[0].lower() == 'germany':
        country = [1,0]
    elif features[0].lower() == 'spain':
        country = [0,1]
    gen = 0
    if features[1].lower() == 'male':
        gen =  1
    country.append(gen)
    features = features[2:]
    features = [int(x) for x in features]
    features+=country
    features = np.array(features)
    df_input = pd.DataFrame([features], columns=main_cols)
    features = ss.transform(df_input)
    print(features)
    res = model.predict(features.reshape(1,-1))

    return render_template('index.html', predicted_value="Customer Churn rate: {}%".format(str(res[0][0]*100)))

if __name__ == '__main__':
    app.run(debug=True)