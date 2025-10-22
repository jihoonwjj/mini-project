#
# Streamlit의 Text와 실행 상태 출력 기능.
#

import streamlit as st
import pandas as pd
import seaborn as sns

df = sns.load_dataset('iris')

st.title("Streamlit의 Text와 실행 상태 출력 기능")

st.header("이것은 *header()* 함수 입니다.")

st.subheader("이것은 *subheader()* 함수 입니다.")

st.caption("이것은 *caption()* 함수 입니다.")

st.text("이것은 text() 함수 입니다")

st.markdown("""
---
### 이것은 markdown() 함수 입니다.

**bold**
__bold__
*italic*
_italic_

| H1 | H2 | H3  |
|:---|:---:|---:|
| a1 | a2 | a3 |
| b1 | b2 | b3 |

![붓꽃](https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg)
---
""")

st.write("이것은 *write()* 함수 입니다.",df.head(5))

st.error("This is an error!")
st.warning("This is a warning!")
st.success("This is a success!")
st.info("This is an info!")
