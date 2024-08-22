#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    math_score = float(request.form['math_score'])
    reading_score = float(request.form['reading_score'])
    writing_score = float(request.form['writing_score'])

    features = np.array([[math_score, reading_score, writing_score]])
    prediction = model.predict(features)
    return f'Predicted race/ethnicity code: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)

