from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__, template_folder='template')
cols = ['age', 'workclass', 'education-num', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
        'capital-loss', 'native-country', 'hour_worked_bins']
data = pd.read_csv('adult-data.txt', sep=",", header=None, names=cols, engine='python')

model = pickle.load(open('AdaBoost_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    le = preprocessing.LabelEncoder()
    data1 = ", ".join(features)
    form_data = open('form.txt', "w")
    form_data.write(data1)
    form_data.close()
    features = np.array(features)
    columns = ['age', 'workclass', 'education-num', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
               'capital-loss', 'native-country', 'hour_worked_bins']
    df = pd.read_csv("form.txt", sep=",\s", header=None, names=columns, engine='python')
    df['hour_worked_bins']=df["hour_worked_bins"].astype('object')
    wdata = pd.concat([data, df])
    wdata.reset_index(inplace=True, drop=True)
    for col1 in set(wdata.columns) - set(wdata.describe().columns):
        wdata[col1] = wdata[col1].astype('category')
    for i in list(wdata.select_dtypes('category')):
        wdata[i] = le.fit_transform(wdata[i])
    prediction = str(model.predict(wdata.tail(1)))

    if prediction[2:6] == ">50K":
        prediction = 1
    else:
        prediction = 0
    print(prediction)
    return render_template('output.html', result=prediction)


if __name__ == '__main__':
    app.run(debug=True)
