# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:38:56 2025

@author: kumar
"""
import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

working_dir = os.path.dirname(os.path.abspath(__file__))

# loading saved model 
path0 = f'{working_dir}/saved_model/scaler_diab.sav'
scaler_f = pickle.load(open(path0,'rb')) 

path_1 = f'{working_dir}/saved_model/trained_model.sav'
diabetes_model = pickle.load(open(path_1,'rb'))

path_2 = f'{working_dir}/saved_model/heart_disease_1.sav'
heart_model = pickle.load(open(path_2,'rb'))

path_3 = f'{working_dir}/saved_model/parkinson_disease.sav'
parkinson_model = pickle.load(open(path_3,'rb'))

path_4 = f'{working_dir}/saved_model/scaler_heart_1.sav'
scaler_heart = pickle.load(open(path_4,'rb'))

path_5 = f'{working_dir}/saved_model/scaler_parkinson.sav'
scaler_park = pickle.load(open(path_5,'rb'))

# sidebar for showing / choosing option 
with st.sidebar:
    selected = option_menu('Multiple Desease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinson Prediction'],
                           icons=['activity'
                                ,'heart',
                                'person']
                           ,default_index=0)
    
# diabetes prediction page 
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction WebPage')
    
    col1 ,col2 ,col3 = st.columns(3)
    
    with col1:
        Pregnancies= st.text_input('Number of preganancies')
    
    with col2:
        Glucose= st.text_input('Glucose level')

    with col3:
        BloodPressure= st.text_input('BloodPressure value')
    
    with col1:
        SkinThickness = st.text_input('SkinThickness')
    with col2:
        Insulin= st.text_input('Insulin level')
    with col3:
        BMI= st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction= st.text_input('Diabetes Pedigree')
    with col2:
        Age= st.text_input('Age')
    
    # code for predicting 
    diab_diagnosis =""
    # creating button for result
    if st.button('Diabetes Test Results'):
        data_val = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        std_data = scaler_f.transform(data_val)
        diab_prediction = diabetes_model.predict(std_data)
        if (diab_prediction[0]==1):
            diab_diagnosis='The Person Has Diabetes'
        else:
            diab_diagnosis='The Person Does Not have Diabetes'
    st.success(diab_diagnosis)             

if (selected == 'Heart Disease Prediction'):
    # page title
    st.title('Heart Disease Prediction WebPage')
    
    col1 ,col2 ,col3 = st.columns(3)
    
    with col1:
        age= st.text_input('Age ')
    
    with col2:
        sex= st.text_input('Sex')

    with col3:
        cp= st.text_input('CP ')
    
    with col1:
        trestbps = st.text_input('trestbps')
    with col2:
        chol= st.text_input('chol')
    with col3:
        fbs= st.text_input('fbs')
    with col1:
        restecg= st.text_input('restecg')
    with col2:
        thalach= st.text_input('thalach')
    with col3:
        exang= st.text_input('exang')
    with col1:
        oldpeak= st.text_input('oldpeak')
    with col2:
        slope= st.text_input('slope')
    with col3:
        ca= st.text_input('ca')
    with col1:
        thal= st.text_input('thal')
    
    # code for predicting 
    diab_diagnosis_2 =""
    # creating button for result
    if st.button('Heart Disease Test Results'):
        data_val_1 = [[float(age), float(sex), float(cp), float(trestbps), float(chol), 
               float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), 
               float(slope), float(ca), float(thal)]]
        
        std_data_1 = scaler_heart.transform(data_val_1)
        heart_predict = heart_model.predict(std_data_1)
        if (heart_predict[0]==1):
            diab_diagnosis_2='The Person Has Heart Problem'
        else:
            diab_diagnosis_2='The Person Does Not have Heart Problem'
    st.success(diab_diagnosis_2)

if (selected == 'Parkinson Prediction'):
    # page title
    st.title('Parkinson Prediction WebPage')
    
    col1 ,col2 ,col3 = st.columns(3)
    
    with col1:
        MDVPFo= st.text_input('MDVP:Fo(Hz) ')
    
    with col2:
        MDVPFhi= st.text_input('MDVP:Fhi(Hz)')

    with col3:
        MDVPFlo= st.text_input('MDVP:Flo(Hz)')
    
    with col1:
        MDVPJitter= st.text_input('MDVP:Jitter(%)')
    with col2:
        MDVPJitter_Abs= st.text_input('MDVP:Jitter(Abs)')
    with col3:
        MDVPRAP= st.text_input('MDVP:RAP')
    with col1:
        MDVPPPQ= st.text_input('MDVP:PPQ')
    with col2:
        JitterDDP= st.text_input('Jitter:DDP')
    with col3:
        MDVPShimmer= st.text_input('MDVP:Shimmer')
    with col1:
        MDVPShimmer_db= st.text_input('MDVP:Shimmer(dB)')
    with col2:
        ShimmerAPQ3= st.text_input('Shimmer:APQ3')
    with col3:
        ShimmerAPQ5= st.text_input('Shimmer:APQ5')
    with col1:
        MDVPAPQ= st.text_input('MDVP:APQ')

    with col2:
        ShimmerDDA= st.text_input('Shimmer:DDA')
    with col3:
        NHR= st.text_input('NHR')
    with col1:
        HNR= st.text_input('HNR')
    with col2:
        RPDE= st.text_input('RPDE')
    with col3:
        DFA= st.text_input('DFA')
    with col1:
        spread1= st.text_input('spread1')
    with col1:
        spread2= st.text_input('spread2')
    with col2:
        D2= st.text_input('D2')
    with col3:
        PPE= st.text_input('PPE')
    
    # code for predicting 
    diab_diagnosis_3 =""
    # creating button for result
    if st.button('Parkinson Disease Prediction Result'):
        data_val_2 = [[MDVPFo,MDVPFhi,MDVPFlo,MDVPJitter,MDVPJitter_Abs,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmer_db,ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]
        
        std_data_2 = scaler_park.transform(data_val_2)
        park_predict = parkinson_model.predict(std_data_2)
        if (park_predict[0]==1):
            diab_diagnosis_3='The Person Has Parkinson'
        else:
            diab_diagnosis_3='The Person Does Not have Parkinson'
    st.success(diab_diagnosis_3)
