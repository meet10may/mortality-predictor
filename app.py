import numpy as np
import pickle
import streamlit as st
import pickle
import pandas as pd
# from flask import Flask,request, jsonify, render_template


def load_pickle(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

# Diabetes_trained_model = pickle.load(open('models/diabetes_trained_model.sav', ‘rb’))
# heart_trained_model = pickle.load(open(‘models/heart_trained_model.sav’, ‘rb’))
# parkinson_trained_model = pickle.load(open(‘models/parkinson_trained_model.sav’, ‘rb’))
# Diabetes_trained_model = load_pickle('model/random_forest_day3_with_imputation.pickle')
# heart_trained_model = load_pickle('model/random_forest_day3_with_imputation.pickle')
# parkinson_trained_model = load_pickle('model/random_forest_day3_with_imputation.pickle')



# Sidebar for Navigation
with st.sidebar:
   st.title("Online calculator for predicting mortality")
   st.write("No input data will be collected from users of this web ARDS mortality prediction tool. It is only meant to make predictions based on the input data.")
#  selected = st.radio('Navigation',
#  selected = option_menu('Multiple Disease Prediction System',
#  ['Diabetes Prediction',
#  'Heart Disease Prediction',
#  'Parkinsons Prediction'],
#  icons=['activity','heart','person'],
#  default_index=0)
 



def predict(df):
    # age = float(request.form.get('age',0))
    # bicarblevel = float(request.form.get('bicarblevel',0))
    # gluclevel = float(request.form.get('gluclevel',0))
    # alblevel = float(request.form.get('alblevel',0))
    # maplevel = float(request.form.get('maplevel',0))
    # hrlevel = float(request.form.get('hrlevel',0))
    # pltlevel = float(request.form.get('pltlevel',0))
    # sex = str(request.form.get('sex',0))
    # is_pneumonia = str(request.form.get('is_pneumonia',0))
    # if sex=='Female': # Male -> 1 and Female -> 2
    #     sex = 2
    # else:
    #     sex = 1
    # if is_pneumonia=='Yes':
    #     is_pneumonia = 1
    # else:
    #     is_pneumonia = 0

    # patient_data = {'Gender':[sex],'Age':[age],'Pneumonia':[is_pneumonia],'Bicarbonate':[bicarblevel],'Glucose':[gluclevel],'Albumin':[alblevel],
    #                 'MAP':[maplevel],'HeartRate':[hrlevel],'Platelet':[pltlevel]}
    # df = pd.DataFrame(patient_data)

    model = load_pickle('model/Random_Forest_Classifier_day_3_with_imputation.pickle')
    st.write(model)
    prob = model.predict_proba(df)
    prob = np.round(prob[0,0,]*100,2)
    data = {'probability': prob}
    # data = jsonify(data)
    return data

# Creating a function for prediction
# def diabetes_prediction(input_data):
#  # changing the input_data to numpy array
#  input_data_as_numpy_array = np.asarray(input_data)


# # reshape the array as we are predicting for one instance
#  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#  scaler = StandardScaler()
#  # standardize the input data
#  std_data = scaler.fit_transform(input_data_reshaped)
#  prediction = Diabetes_trained_model.predict(std_data)

#  if (prediction[0] == 0):
#     return "The person is not diabetic"
#  else:
#     return "The person is diabetic"
 
def main():
 # Giving Title
 st.title('Predicting mortality for ARDS')
 st.write("Please fill in the patient’s information (assessed on day 3 after ARDS criteria are met) to compute the probability of mortality (all fields are required)")
 # Getting input data from the user
 with st.form(key='columns_in_form'):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Patient's Information**")
        age = st.number_input("Age (years):",step =1)
        sex = st.radio("Sex:",["Male","Female"])
    with c2:
        st.write("**Clinical Information**")
        bicarblevel = st.number_input("Enter Bicarbonate level (mmol/L):",min_value=0.0)
        gluclevel = st.number_input("Enter Glucose level (mg/dl):",min_value=0.0)
        alblevel = st.number_input("Enter Albumin level(g/dl):",min_value=0.0)
        maplevel = st.number_input("Enter Mean airway pressure (cmH20):",min_value=0.0)
        hrlevel = st.number_input("Enter Heart rate(bpm):",min_value=0.0)
        pltlevel = st.number_input("Enter Platelet count(10^9/L):",min_value=0.0)
        is_pneumonia = st.radio("Pneumonia:",['Yes','No'])
    

    submitButton = st.form_submit_button(label = 'Make prediction')
    if submitButton:
        if sex=='Female': # Male -> 1 and Female -> 2
            sex = 2
        else:
            sex = 1
        if is_pneumonia=='Yes':
            is_pneumonia = 1
        else:
            is_pneumonia = 0

        patient_data = {'Gender':[sex],'Age':[age],'Pneumonia':[is_pneumonia],'Bicarbonate':[bicarblevel],'Glucose':[gluclevel],'Albumin':[alblevel],
                        'MAP':[maplevel],'Heart rate':[hrlevel],'Platelets':[pltlevel]}
        df = pd.DataFrame(patient_data)
        
        data = predict(df)
        if data['probability'] < 50:
            st.error("The chance of survival is: {}%".format(data['probability']))
        else:
            st.success("The chance of survival is: {}%".format(data['probability']))

 # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
#  Pregnancies = st.text_input('Number of Pregnancies')
#  Glucose = st.text_input('Glucose Level')
#  BloodPressure = st.text_input('Blood Pressure Level')
#  SkinThickness = st.text_input('Skin Thickness Value')
#  Insulin = st.text_input('Insulin Level')
#  BMI = st.text_input('BMI Value')
#  DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
#  Age = st.text_input('Age of the Person')
 
 # Code for prediction
#  diagnosis = ''
 
 # Creating button for Prediction
#  if st.button('Diabetes Test Result'):
#     diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
 
#  st.success(diagnosis)
 
if __name__ == '__main__':
 main()

