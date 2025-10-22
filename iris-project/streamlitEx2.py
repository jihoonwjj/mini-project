#
# Streamlit의 시각화 기능.
#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Streamlit의 시각화 기능")

my_data = np.random.randn(30,3)
df = pd.DataFrame(data = my_data , columns=['a', 'b', 'c'])

st.write("랜덤으로 생성된 데이터:", df.head())

st.header("1. line_chart() 예시:")
st.line_chart(df)

st.header("2. area_chart() 예시:")
st.area_chart(df)

st.header("3. bar_chart() 예시:")
st.bar_chart(df)

st.header("4. pyplot() 예시:")
df_iris = sns.load_dataset("iris")
df_iris_setosa = df_iris[df_iris.species=="setosa"]
df_iris_versicolor = df_iris[df_iris.species=="versicolor"]
df_iris_virginica = df_iris[df_iris.species=="virginica"]

my_fig, my_ax = plt.subplots()
my_ax.scatter(df_iris_setosa.sepal_length, df_iris_setosa.sepal_width, color="green", alpha=0.5, label="Setosa")
my_ax.scatter(df_iris_versicolor.sepal_length, df_iris_versicolor.sepal_width, color="blue", alpha=0.5, label="Versoicolor")
my_ax.scatter(df_iris_virginica.sepal_length, df_iris_virginica.sepal_width, color="red", alpha=0.5, label="Virginica")
my_ax.legend(loc="lower right")
my_ax.set_xlabel("Sepal Length")
my_ax.set_ylabel("Sepal Width")
my_ax.set_title("Matplotlib Visualization")
st.pyplot(my_fig)



