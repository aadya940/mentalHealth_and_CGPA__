from textwrap import fill
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import *
import pickle
import mixed_naive_bayes


def preprocess_input(my_dataframe):
    replacements = [{'Female': 0, 'Male': 1},
        {18.0: 0, 21.0: 1, 19.0: 2, 22.0: 3, 23.0: 4, 20.0: 5, 24.0: 6},
        {'Engineering': 0,
        'Islamic education': 1,
        'BIT': 2,
        'Laws': 3,
        'Mathemathics': 4,
        'Pendidikan islam': 5,
        'BCS': 6,
        'Human Resources': 7,
        'Irkhs': 8,
        'Psychology': 9,
        'KENMS': 10,
        'Accounting ': 11,
        'ENM': 12,
        'Marine science': 13,
        'KOE': 14,
        'Banking Studies': 15,
        'Business Administration': 16,
        'Law': 17,
        'KIRKHS': 18,
        'Usuluddin ': 19,
        'TAASL': 20,
        'Engine': 21,
        'ALA': 22,
        'Biomedical science': 23,
        'koe': 24,
        'Kirkhs': 25,
        'BENL': 26,
        'Benl': 27,
        'IT': 28,
        'CTS': 29,
        'engin': 30,
        'Econs': 31,
        'MHSC': 32,
        'Malcom': 33,
        'Kop': 34,
        'Human Sciences ': 35,
        'Biotechnology': 36,
        'Communication ': 37,
        'Diploma Nursing': 38,
        'Pendidikan Islam ': 39,
        'Radiography': 40,
        'psychology': 41,
        'Fiqh fatwa ': 42,
        'DIPLOMA TESL': 43,
        'Koe': 44,
        'Fiqh': 45,
        'Islamic Education': 46,
        'Nursing ': 47,
        'Pendidikan Islam': 48},
        {'year 1': 0,
        'year 2': 1,
        'Year 1': 2,
        'year 3': 3,
        'year 4': 4,
        'Year 2': 5,
        'Year 3': 6},
        {'No': 0, 'Yes': 1},
        {'Yes': 0, 'No': 1},
        {'No': 0, 'Yes': 1},
        {'Yes': 0, 'No': 1},
        {'No': 0, 'Yes': 1}]
    for i, j in zip(my_dataframe.columns, replacements):
        my_dataframe[i] = my_dataframe[i].replace(j)
    return my_dataframe

def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

def predict(model, X):
    preds = model.predict(X)
    preds = list(preds)
    if preds[0] == 0 :
        preds[0] = '3.00 - 3.49'
    if preds[0] == 1 :
        preds[0] = '3.50 - 4.00'
    if preds[0] == 2:
        preds[0] = '3.50 - 4.00'
    if preds[0] == 3:
        preds[0] = '2.50 - 2.99'
    if preds[0] == 4:
        preds[0] = '2.00 - 2.49'
    if preds[0] == 5:
        preds[0] = '0 - 1.99'

    return "Your GPA is forecasted to be in range : {}".format(preds[0])


def main():
    df = pd.read_csv(r'C:\Users\Aadya\Desktop\mental_health.csv')
    df = df.drop('Timestamp', axis=1)

    st.title("Analysis of Mental Health and Studies")
    st.subheader("by Aadya, Aman, Shaan, Soham")
    st.subheader("-------------------------------------------------------------------------------")
    with st.form(key="form1"):
        gender_ = st.selectbox("Gender", options=["Male", "Female"])
        age_ = st.number_input("Your Age", min_value= 0 , max_value= 100)

        courses = ['Engineering', 'Islamic education', 'BIT', 'Laws', 'Mathemathics',
       'Pendidikan islam', 'BCS', 'Human Resources', 'Irkhs',
       'Psychology', 'KENMS', 'Accounting ', 'ENM', 'Marine science',
       'KOE', 'Banking Studies', 'Business Administration', 'Law',
       'KIRKHS', 'Usuluddin ', 'TAASL', 'Engine', 'ALA',
       'Biomedical science', 'koe', 'Kirkhs', 'BENL', 'Benl', 'IT', 'CTS',
       'engin', 'Econs', 'MHSC', 'Malcom', 'Kop', 'Human Sciences ',
       'Biotechnology', 'Communication ', 'Diploma Nursing',
       'Pendidikan Islam ', 'Radiography', 'psychology', 'Fiqh fatwa ',
       'DIPLOMA TESL', 'Koe', 'Fiqh', 'Islamic Education', 'Nursing ',
       'Pendidikan Islam']

        course_ = st.selectbox("What is your course?", options = courses)
        year_of_study = st.selectbox("Your current year of study?", options = ['year 1', 'year 2',  'year 3', 'year 4'])
        marital_status = st.selectbox("Your marital status?", options= ['No', 'Yes'])
        depression_ = st.selectbox("Do you have depression?", options = ['No', 'Yes'])
        anxiety_ = st.selectbox("Do you have anxiety?", options=['No', 'Yes'])
        panic_ = st.selectbox("Do you have panic attacks?", options = ['Yes', 'No'])
        treatment_ = st.selectbox("Did you seek any specialist's treatment?", options=['Yes', 'No'])

        features = [gender_, age_, course_, year_of_study,
        marital_status, depression_, anxiety_, panic_, treatment_ ]

        submit_button = st.form_submit_button(label="Analyse")

    if submit_button:
        my_dict= {}
        for i, j in zip(df.drop('What is your CGPA?', axis = 1).columns, features):
            my_dict[i] = j
        input_values = pd.DataFrame(my_dict, index=[0])
        X = preprocess_input(input_values)
        model = load_model(r"C:\Users\Aadya\Desktop\my_finalized_model.sav")
        my_pred = predict(model, X)
        # st.write(input_values.columns)
        st.write(my_pred)

        if my_dict['Do you have Depression?'] == 'Yes' :
            st.subheader('Here are some ways to cope with Depression')
            st.write("Don't withdraw from life. Socialising can improve your mood.",
            "Be more active.",
            "Face your fears.",
            "Try to eat a healthy diet.",
            "Have a routine.",
            "Seek medical Help.")
        
        if my_dict['Do you have Anxiety?'] == 'Yes' :
            st.subheader('Here are some ways to deal with axiety')
            st.write('Use stress management and relaxation techniques.',
            'Make sleep a priority.',
            'Learn about your disorder.',
            'Keep physically active.')
        
        if my_dict['Do you have Panic attack?'] == 'Yes':
            st.subheader('Here are a few Quick tips to help reduce panic attacks')
            st.write('Doing breathing exercises every day will help to prevent panic attacks and relieve them when they are happening.',
            'Regular exercise, especially aerobic exercise, will help you to manage stress levels, release tension, improve your mood and boost confidence.',
            'Eat regular meals to stabilise your blood sugar levels.',
            'Avoid caffeine, alcohol and smoking – these can make panic attacks worse.',
            'Cognitive behavioural therapy (CBT) can identify and change the negative thought patterns that are feeding your panic attacks.')
        



if __name__ == '__main__':
    main()
