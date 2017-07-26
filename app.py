__author__ = 'c4'
from flask import Flask
from flask import render_template
from flask import request
from flask_bootstrap import Bootstrap
from sklearn.externals import joblib
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
Bootstrap(app)

scaler, classifier = joblib.load('diabetes.pkl')

def predict(x):
    x = np.asarray(x)
    x = x.astype(float)
    x = x.reshape(1,-1)
    x = scaler.transform(x)
    y = classifier.predict(x)
    y = int(y)
    return y

labels = [('Pregnancies', 0, 17),
          ('Glucose', 55, 200),
          ('Blood Pressure (Diastolic)', 25, 110),
          ('SkinThickness', 5, 100),
          ('Insulin', 20, 850),
          ('BMI', 10, 70),
          ('Diabetes Pedigree Function (Genetic Influence)', 0, 2),
          ('Age', 21, 80)]

@app.route('/')
def index():
    predictors = []
    features = []
    for i in range(0, 8):
        features.append(labels[i][0])

    for feature in features:
        predictors.append(request.args.get(feature))

    if all(predictors):
        features[:] = []
        prediction = predict(predictors)
        if prediction == 1:
            message = 'Patient has diabetes.'
        else:
            message = 'Patient doesn\'t have diabetes.'
        predictors[:] = []
        predicted = 1
        return render_template('index.html', labels=labels, predicted=predicted, prediction=prediction, message=message)
    else:
        prediction = 0
        predicted = 0
        message = 'Enter all values.'
        return render_template('index.html', labels=labels, predicted=predicted, prediction=prediction, message=message)

if __name__ == '__main__':
    app.run()


