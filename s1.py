import numpy as np
import streamlit as st
import pandas as pd
import joblib

# print(r2_score(ytr,pipe1.predict(xtr)))


st.title("CAR price predictor")
car=pd.read_csv('Cleaned_car.csv')

# model =pickle.load(open('LinearRegressionModel.pkl','rb'))
# with open("gradientboosting.pkl", "rb") as f:
#     model = pickle.load(f)

# model =joblib.load('car_price_predictor_pipeline.joblib')
companies=sorted(car['company'].unique())
# name =sorted(car[car['company'==a]]['name'].unique())
a =st.selectbox("choose car company",companies)
name =sorted(car[car['company']==a]['name'].unique())


b=st.selectbox("choose car model",name)

c=st.selectbox("chose fuel type",['Petrol','Diesel'])

d=st.number_input("enter year",1995,2019)

e=st.number_input("kms driven",0,400000)
inp = pd.DataFrame({
    'name': [b],
    'company': [a],
    'year': [int(d)],  # ← explicitly cast to int
    'kms_driven': [int(e)],  # ← explicitly cast to int
    'fuel_type': [c]
})

pipe1=joblib.load('car.joblib')

if st.button('predict'):
    res=pipe1.predict(inp)
    st.write(f"__car price is {res} INR__")

# print
def set_bg(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# set_bg("https://marketplace.canva.com/4qKkY/MAEfeh4qKkY/1/tl/canva-white-honda-car-parked-on-road-MAEfeh4qKkY.jpg")
