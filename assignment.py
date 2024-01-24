from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


iris = load_iris()
X, y = iris.data, iris.target


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features = [float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])]

    
    prediction = clf.predict([features])[0]

    
    species_name = iris.target_names[prediction]

    return render_template('result.html', species=species_name)

if __name__ == '__main__':
    app.run(debug=True)