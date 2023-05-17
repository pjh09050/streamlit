import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 페이지 너비 조정 
st.set_page_config(
    page_title="기말 프로젝트",
    page_icon="heart",
    layout="wide",
)
st.markdown(
    """
    <style>
    .main {
        max-width: 1300px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
############################################################################
introduce = st.sidebar.checkbox('프로젝트 소개')
if introduce == True:
    st.title('주제')
    st.write('\n')
    # header
    st.header('예시 데이터')
    st.write('\n')
    # subheader
    st.subheader('데이터 소개 및 진행 방법')
    st.write('\n')
    # layout(checkbox 만들기)
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    cl1 = col1.checkbox('EDA 사용법')
    st.write('\n')
    if cl1 == True:
        st.subheader('EDA 종류')
        st.write('correlation은 이런걸 확인할 수 있음 등등')
    cl2 = col2.checkbox('방법론 1 사용법')
    st.write('\n')
    if cl2 == True:
        st.subheader('방법론 1 사용법')
        st.write('방법론 1은 이렇게 사용하여 이런 결과를 볼 수 있음')
    cl3 = col3.checkbox('방법론 2 사용법')
    st.write('\n')
    if cl3 == True:
        st.subheader('방법론 2 사용법')
        st.write('방법론 2는 이렇게 사용하여 이런 결과를 볼 수 있음')
    cl4 = col4.checkbox('결론 도출 방법')
    st.write('\n')
    if cl4 == True:
        st.subheader('결론 도출 방법')
        st.write('이런과정으로 결론을 이렇게 도출할 수 있음')
    st.write('\n')

    st.subheader('프로젝트 소개와 분석 방법, 이것저것 소개')
############################################################################

st.sidebar.header('데이터 불러오기')
df = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if df is not None:
    try:
        data = pd.read_csv(df)
    except:
        data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/final_df.csv"
        data = pd.read_csv(data_url)
else:
    data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/final_df.csv"
    data = pd.read_csv(data_url)


with st.sidebar:
    choose = option_menu("데이터 확인", ["EDA", "방법론1", "방법론2"],
                         icons=['bar-chart', 'bar-chart', 'bar-chart'],
                         menu_icon="bi bi-card-list", default_index=0,
                         styles={
                         # default_index = 처음에 보여줄 페이지 인덱스 번호
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    } # css 설정
    )

if choose == 'EDA' and introduce == False:
    st.subheader('1. 데이터 확인')
    data_check = st.checkbox('파일 데이터 확인')
    if data_check:
        st.write('로드된 데이터와 크기:', data.shape)
        st.write(data)
    df_col = data.columns

    st.subheader('2. 변수 확인')
    species_graph = st.checkbox('변수 확인 그래프 그리기')
    if species_graph:
        select_multi_species = st.selectbox('확인하고 싶은 변수 선택',df_col)
        if select_multi_species:
            st.subheader(select_multi_species + ' 그래프')
            select = data[select_multi_species]
            fig = px.bar(select, x=data['datetime'], y=select.values)
            fig.update_layout(width=1100, height=500)
            st.plotly_chart(fig)         
        else:
            st.write('선택된 변수가 없습니다.')

    st.subheader('3. Correlation')
    select_heatmap = st.checkbox('Heatmap 그리기')
    if select_heatmap:
        mask = np.zeros_like(data.corr(), dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(10,4))
        sns.heatmap(data = data.corr(), annot=True, mask = mask,
                    fmt = '.2f', linewidths=.5, cmap='Blues')
        st.pyplot(plt.gcf())

    st.subheader('4. VIF')
    check_vif = st.checkbox('VIF 확인')
    if check_vif:
        df2 = data.drop(['datetime'], axis=1)
        vif = pd.DataFrame()
        vif["features"] = df2.columns 
        vif["VIF Factor"] = [variance_inflation_factor(df2.values, i) for i in range(df2.shape[1])]
        vif = vif.sort_values(by="VIF Factor", ascending=False)
        vif = vif.reset_index().drop(columns='index')
        vif
        

if choose == '방법론1' and introduce == False:
    st.subheader('선택한 방법론 : 방법론1')
    st.sidebar.header('파라미터 튜닝')
    # 첫 번째 파라미터를 입력받는 텍스트 상자
    param1 = st.sidebar.text_input('첫 번째 파라미터', '')
    # 두 번째 파라미터를 입력받는 텍스트 상자
    param2 = st.sidebar.text_input('두 번째 파라미터', '')
    # 세 번째 파라미터를 입력받는 슬라이더
    param3 = st.sidebar.slider('세 번째 파라미터', 0, 100, 50)

    # 입력 받은 파라미터 값을 출력
    st.write('첫 번째 파라미터:', param1)
    st.write('두 번째 파라미터:', param2)
    st.write('세 번째 파라미터:', param3)

if choose == '방법론2' and introduce == False:
    st.subheader('선택한 방법론 : 방법론2')
    st.sidebar.header('파라미터 튜닝')
    # 첫 번째 파라미터를 입력받는 텍스트 상자
    param1 = st.sidebar.text_input('첫 번째 파라미터', '')
    # 두 번째 파라미터를 입력받는 텍스트 상자
    param2 = st.sidebar.text_input('두 번째 파라미터', '')
    # 세 번째 파라미터를 입력받는 슬라이더
    param3 = st.sidebar.slider('세 번째 파라미터', 0, 100, 50)

    # 입력 받은 파라미터 값을 출력
    st.write('첫 번째 파라미터:', param1)
    st.write('두 번째 파라미터:', param2)
    st.write('세 번째 파라미터:', param3)