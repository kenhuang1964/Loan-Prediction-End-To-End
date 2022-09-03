import joblib
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session, escape
import numpy as np
import pandas as pd
import utils
from utils import preprocessdata

@app.route('/') 
def home(): 
    return render_template('index.html') 
@app.route('/predict/', methods=['GET', 'POST'])

def predict():  
    if request.method == 'POST': 
        Gender = request.form.get('Gender')
        Married = request.form.get('Married')
        Dependents = request.form.get('Dependents')
        Education = request.form.get('Education')
        Self_Employed = request.form.get('Self_Employed')  
        ApplicantIncome = request.form.get('ApplicantIncome')  
        CoapplicantIncome = request.form.get('CoapplicantIncome') 
        LoanAmount = request.form.get('LoanAmount')   
        Loan_Amount_Term = request.form.get('Loan_Amount_Term')   
        Credit_History = request.form.get('Credit_History')   
        Property_Area = request.form.get('Property_Area')  

    prediction = utils.preprocessdata(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
       CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
       Property_Area)

    return render_template('predict.html', prediction=prediction) 

if __name__ == '__main__': 
    app.run(debug=True) 