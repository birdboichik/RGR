import pandas as pd 
import numpy as np 
import sklearn
import pickle
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, rand_score
from keras.models import load_model
import io
import sklearn

modelsvm = pickle.load(open('models/kekshrek.pkl', 'rb'))
modelbagging = pickle.load(open('models/bag.pkl', 'rb'))
modelkmeans = pickle.load(open('models/kmean.pkl', 'rb'))
modelkgradient = pickle.load(open('models/GradientBoost.pkl', 'rb'))
modelstacking = pickle.load(open('models/Stack.pkl', 'rb'))
modelneurall = load_model('models/Neurall.h5')
data = pd.read_csv('card_transdata_upd.csv')
X = data.drop('fraud', axis=1)
y = data['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("Предсказания моделей машинного обучения")

uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

if uploaded_file is None:
    st.subheader("Введите данные:")

    input_data = {}
    feature_names = ['Unnamed: 0','distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    for feature in feature_names:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=10.0)

    if st.button('Сделать предсказание'):

        input_df = pd.DataFrame([input_data])

        st.write("Входные данные:", input_df)

        # Делаем предсказания
        prediction_ml1 = modelsvm.predict(input_df)
        prediction_ml2 = modelkmeans.predict(input_df)
        prediction_ml3 = modelstacking.predict(input_df)
        prediction_ml4 = modelkgradient.predict(input_df)
        prediction_ml5 = modelbagging.predict(input_df)
        prediction_ml6 = (modelneurall.predict(input_df) > 0.5).astype(int)

        # Вывод результатов
        st.success(f"Результат предсказания SVC: {prediction_ml1}")
        st.success(f"Результат предсказания kmeans.pkl: {prediction_ml2}")
        st.success(f"Результат предсказания Stacking: {prediction_ml3}")
        st.success(f"Результат предсказания GradientBoosting: {prediction_ml4}")
        st.success(f"Результат предсказания Bagging: {prediction_ml5}")
        st.success(f"Результат предсказания Neural: {prediction_ml6}")
else:
    data1 = pd.read_csv(uploaded_file)
    data1
    data1.columns

    st.number_input()

        
    