import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 페이지 설정
st.set_page_config(
    page_title="MPG 연비 예측 앱",
    page_icon="🚗",
    layout="wide"
)

# 제목
st.title("🚗 MPG 연비 예측 App")

# 사이드바에 입력 폼
st.sidebar.header("차량의 SPEC을 입력해 주세요.")

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    """데이터를 로드하고 전처리합니다."""
    # Seaborn의 mpg 데이터셋 로드
    df = sns.load_dataset("mpg")
    
    # name 컬럼 제거
    df = df.drop(columns="name")
    
    # 결측치 제거
    df = df.dropna()
    
    # origin을 더미 변수로 변환
    df = df.join(pd.get_dummies(df["origin"], drop_first=True)).drop(columns=["origin"])
    
    return df

@st.cache_resource
def train_model():
    """모델을 학습하고 저장합니다."""
    df = load_and_preprocess_data()
    
    # X, y 분리
    X = df.drop(columns="mpg")
    y = df["mpg"]
    
    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # RandomForest 모델 학습
    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist()

# 모델과 스케일러 로드
model, scaler, feature_names = train_model()

# 입력 위젯들
st.sidebar.subheader("차량 사양 입력")

# 수치형 변수들 (slider)
cylinders = st.sidebar.slider(
    "cylinders", 
    min_value=3, 
    max_value=8, 
    value=7, 
    step=1
)

displacement = st.sidebar.slider(
    "displacement", 
    min_value=68.0, 
    max_value=455.0, 
    value=88.37, 
    step=0.01
)

horsepower = st.sidebar.slider(
    "horsepower", 
    min_value=46.0, 
    max_value=230.0, 
    value=62.34, 
    step=0.01
)

weight = st.sidebar.slider(
    "weight", 
    min_value=1613.0, 
    max_value=5140.0, 
    value=1949.46, 
    step=0.01
)

acceleration = st.sidebar.slider(
    "acceleration", 
    min_value=8.0, 
    max_value=24.8, 
    value=10.82, 
    step=0.01
)

model_year = st.sidebar.slider(
    "model_year", 
    min_value=70, 
    max_value=82, 
    value=72, 
    step=1
)

# 명목형 변수 (selectbox)
origin = st.sidebar.selectbox(
    "origin",
    options=["Europe", "Japan", "USA"],
    index=0
)

# origin을 더미 변수로 변환
japan = 1 if origin == "Japan" else 0
usa = 1 if origin == "USA" else 0

# 입력 데이터 구성
input_data = {
    'cylinders': cylinders,
    'displacement': displacement,
    'horsepower': horsepower,
    'weight': weight,
    'acceleration': acceleration,
    'model_year': model_year,
    'japan': japan,
    'usa': usa
}

# 입력 데이터를 DataFrame으로 변환
input_df = pd.DataFrame([input_data])

# 예측을 위한 데이터 준비 (mpg 제외한 컬럼들만)
X_input = input_df[feature_names]

# MinMaxScaler 적용
X_input_scaled = scaler.transform(X_input)

# 예측 수행
prediction = model.predict(X_input_scaled)[0]

# 메인 컨텐츠 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("입력된 X 데이터:")
    st.dataframe(X_input, use_container_width=True)
    
    st.subheader("전처리된 X 데이터:")
    X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=feature_names)
    st.dataframe(X_input_scaled_df, use_container_width=True)

with col2:
    st.subheader("예측 결과:")
    st.metric(
        label="MPG 예측값",
        value=f"{prediction:.2f}",
        delta=None
    )
    
    # MPG 분포 히스토그램
    st.subheader("MPG Distribution and Yours in Red")
    
    # 원본 데이터 로드
    df = load_and_preprocess_data()
    
    # 히스토그램 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['mpg'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    
    # 예측값에 빨간색 점선 추가
    ax.axvline(x=prediction, color='red', linestyle='--', linewidth=2, 
               label=f'Your Prediction: {prediction:.2f}')
    
    ax.set_xlabel('mpg')
    ax.set_ylabel('Frequency')
    ax.set_title('MPG Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# 추가 정보
st.subheader("📊 데이터 정보")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("총 데이터 수", len(load_and_preprocess_data()))

with col2:
    st.metric("평균 MPG", f"{load_and_preprocess_data()['mpg'].mean():.2f}")

with col3:
    st.metric("표준편차", f"{load_and_preprocess_data()['mpg'].std():.2f}")

# 모델 성능 정보
st.subheader("🤖 모델 정보")
st.info("이 앱은 RandomForest Regressor를 사용하여 연비를 예측합니다.")
