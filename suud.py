import pandas as pd
import numpy as np
import pickle
import streamlit as st
import sklearn
# clear warning
import warnings
warnings.filterwarnings("ignore")

loaded_model = pickle.load(open('NaiveBayesModel.sav', 'rb'))

# Caching the model for faster loading


# @st.cache
# imp = np.array([[5.0,	3.6,	1.4,	0.2]])
# y_pred = model.predict(imp)
# print(y_pred)
st.title('Deploy Model')
st.header('Predict Iris Data Menggunakan Naive Bayes')
sepal_length = st.number_input('Sepal Length', min_value=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.1)
petal_length = st.number_input('Petal Length', min_value=0.1)
petal_width = st.number_input('Petal Width', min_value=0.1)

dataInput = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
if st.button('Predict'):
    y_pred = loaded_model.predict(dataInput)
    st.success(f'Predict {y_pred[0]}')
