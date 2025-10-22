#
# Streamlit의 Sidebar 기능과 시각화 App.
#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Streamlit의 Sidebar 기능과 시각화 App")

st.sidebar.header("변수 선택")
my_select_X = st.sidebar.selectbox("X 변수를 선택 하시오:", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
my_select_Y = st.sidebar.selectbox("Y 변수를 선택 하시오:", ["sepal_length", "sepal_width", "petal_length", "petal_width"])
my_select_species = st.sidebar.multiselect("붓꽃 유형을 선택 하시오 (multi):",["setosa", "versicolor", "virginica"])

st.sidebar.header("출력 커스터마이징")
my_alpha = st.sidebar.slider("alpha:", 0.1, 1.0, 0.5)

df_iris = sns.load_dataset("iris")
my_colors = {"setosa":"red", "virginica":"green", "versicolor":"blue"}

if my_select_species:
    my_fig, my_ax = plt.subplots()
    for a_species in my_select_species:
        my_df_select = df_iris[df_iris.species==a_species]
        my_ax.scatter(my_df_select[my_select_X], my_df_select[my_select_Y], color=my_colors[a_species], alpha=my_alpha, label=a_species)
    my_ax.legend(loc="lower right")
    my_ax.set_xlabel(my_select_X)
    my_ax.set_ylabel(my_select_Y)
    my_ax.set_title("Iris Scatter Plot")
    st.pyplot(my_fig)
else:
    st.warning("붓꽃 유형을 선택해 주세요!!!")
