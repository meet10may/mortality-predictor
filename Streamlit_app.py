import numpy as np
import pickle
import streamlit as st
import pandas as pd


def load_pickle(filename):
    model = pickle.load(open(filename, 'rb'))
    return model


# Sidebar for Navigation
with st.sidebar:
   st.title("Online calculator for predicting mortality")
   st.write("No input data will be collected from users of this web ARDS mortality prediction tool. It is only meant to make predictions based on the input data.")

def predict(df):

    # model = load_pickle('./model/Random_Forest_Classifier_day_3_with_imputation.pickle')
    model = pickle.load(open('Random_Forest_Classifier_day_3_with_imputation.pickle','rb'))
    
    # model = load_pickle('Random_Forest_Classifier_day_3_with_imputation.pickle')
    # st.write(model)
    prob = model.predict_proba(df)
    prob = np.round(prob[0,0,]*100,2)
    data = {'probability': prob}
    return data
 
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
 
if __name__ == '__main__':
 main()

