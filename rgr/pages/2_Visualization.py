import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("rgr/photo.jpg")

st.header("Гистограммы")
columns = ['distance_from_home','ratio_to_median_purchase_price','distance_from_last_transaction']

for col in columns:
    plt.figure(figsize=(8, 6))
  
    xlim = (data[col].min(), data[col].mean())
      
    sns.histplot(data.sample(5000)[col], bins=1000)
    plt.title(f'Гистограмма для {col}')
    plt.xlim(xlim)
    st.pyplot(plt)


plt.figure(figsize=(8, 6))
plt.title(f'Тепловая карта')
sns.heatmap(data=data.corr(numeric_only=True), annot=True)
st.pyplot(plt)