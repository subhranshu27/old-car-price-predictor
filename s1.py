import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")

@st.cache_data
def load_data():
    car = pd.read_csv('Cleaned_car.csv')
    car.drop(columns=[col for col in car.columns if 'Unnamed' in col], inplace=True)
    return car

@st.cache_resource
def train_model():
    car = pd.read_csv('Cleaned_car.csv')
    car.drop(columns=[col for col in car.columns if 'Unnamed' in col], inplace=True)

    x = car.drop(columns='Price')
    y = car['Price']
    xtr, _, ytr, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    ohe = OneHotEncoder()
    ohe.fit(x[['name', 'company', 'fuel_type']])

    trf = ColumnTransformer(
        [('ohe_cols', OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type'])],
        remainder='passthrough'
    )
    clf = GradientBoostingRegressor(
        min_impurity_decrease=0.05,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1
    )
    pipe = Pipeline(steps=[('trf', trf), ('clf', clf)])
    pipe.fit(xtr, ytr)
    return pipe

# Load data and train once (cached after first run)
car = load_data()

with st.spinner("Loading model... please wait ⏳"):
    pipe1 = train_model()

# UI
companies = sorted(car['company'].unique())
a = st.selectbox("Choose car company", companies)

names = sorted(car[car['company'] == a]['name'].unique())
b = st.selectbox("Choose car model", names)

c = st.selectbox("Choose fuel type", ['Petrol', 'Diesel'])
d = st.number_input("Enter year", min_value=1995, max_value=2019, value=2015)
e = st.number_input("Kilometres driven", min_value=0, max_value=400000, value=30000)

if st.button('Predict Price'):
    inp = pd.DataFrame({
        'name': [b],
        'company': [a],
        'year': [int(d)],          # ← explicitly cast to int
        'kms_driven': [int(e)],    # ← explicitly cast to int
        'fuel_type': [c]
    })
    res = pipe1.predict(inp)
    st.success(f"💰 Estimated Car Price: ₹ {int(res[0]):,} INR")
