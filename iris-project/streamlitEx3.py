#
# Streamlit의 Widget 기능.
#

import streamlit as st
import pandas as pd

st.title("Streamlit의 Widget 기능")
st.write("---")

val_A = st.slider("A의 값을 설정 하시오.", 0, 10, 5)
st.write("입력된 A의 값 : ",val_A)
st.write("---")

my_click = st.button('Click ME!')        # Reset이 않됨???
st.write("버튼 클릭 여부: ", my_click)   
st.write("---") 

a_checked = st.checkbox("a")
b_checked = st.checkbox("b")
c_checked = st.checkbox("c")

my_df = pd.DataFrame(data={"a":a_checked, "b":b_checked, "c":c_checked}, index=[0])
st.write(my_df)
st.write("---")

my_choice = st.radio("다음 중 한가지를 선택 하시오.", ["Choice 1", "Choice 2", "Choice 3", "Choice 4"])
st.write("나의 선택 값 : ", my_choice)
st.write("---")

my_select = st.selectbox("다음 중 한가지를 선택 하시오.", ["Select 1", "Select 2", "Select 3", "Select 4"])
st.write("나의 선택 값 : ", my_select)
st.write("---")

my_multi_select = st.multiselect("다음 중 한가지 이상을 선택 하시오.",["Choice 1", "Choice 2", "Choice 3", "Choice 4"])
st.write("나의 다중 선택 : ", ", ".join(my_multi_select))
st.write("---")

my_text = st.text_input("문자열 입력:")
st.write("입력된 문자열  : ", my_text)
st.write("---")

my_number = st.number_input("수치 자료 입력:")
st.write("입력된 수치 자료 : ", my_number)
st.write("---")



