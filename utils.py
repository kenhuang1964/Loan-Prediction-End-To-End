import joblib 


def preprocessdata(Gender, Married, Education, Self_Employed, ApplicantIncome,
       CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
       Property_Area):
    test_data = [[Gender, Married, Education, Self_Employed, ApplicantIncome,
       CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
       Property_Area]]  
    trained_model = joblib.load("model.pkl") 
    scaler_model = joblib.load("scaling.pkl")
    new_data = scaler_model.fit_transform(test_data)
    prediction = trained_model.predict(new_data) 

    return prediction 