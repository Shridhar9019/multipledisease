import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

# Prepare data
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'diabetes_model.sav', 'rb'))
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex (1 = male, 0 = female)')

    with col3:
        cp = st.text_input('Cigarettes Per Day')

    with col1:
        totalcol = st.text_input('Total Cholesterol')

    with col2:
        sysbp = st.text_input('Systolic BP')

    with col3:
        glu = st.text_input('Glucose')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            user_input = np.array([[float(age), float(sex), float(cp), float(totalcol), float(sysbp), float(glu)]])
            inputs = scaler.transform(user_input)
            heart_prediction = logreg.predict(inputs)

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'

        except ValueError:
            heart_diagnosis = 'Please enter valid numerical values in all fields.'

    st.success(heart_diagnosis)
