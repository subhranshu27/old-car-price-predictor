import numpy as np
import streamlit as st
import pickle
import pandas as pd
import joblib
# import scikit-learn==1.9.0
import sklearn

car=pd.read_csv('Cleaned_car.csv')
car.drop(columns=['Unnamed: 0'],inplace=True)
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor


x=car.drop(columns='Price')
y=car['Price']
xtr,xte,ytr,yte =train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.metrics import r2_score

ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


trf = ColumnTransformer([('ohe_cols', OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type'])], remainder='passthrough')

clf1 =GradientBoostingRegressor(min_impurity_decrease=0.05,n_estimators=200,max_depth=5,learning_rate=.1)

pipe1 =Pipeline(steps=[('trf',trf),('clf',clf1)])

pipe1.fit(xtr,ytr)

# print(r2_score(ytr,pipe1.predict(xtr)))


st.title("CAR price predictor")


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

inp =pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                  data=np.array([b,a,d,e,c]).reshape(1,5))
inp1=inp.astype(float)

if st.button('predict'):
    res=pipe1.predict(inp1)
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
