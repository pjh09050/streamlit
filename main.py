import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.express as px
import os
# Create a custom HTML/CSS template
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
    .sidebar.sidebar-content {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .sidebar .sidebar-content .element-container:last-child {
        order: -1;
        margin-top: auto;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
############################################################################
introduce = st.sidebar.checkbox('프로젝트 소개', value=False)
if introduce == True:
    st.title('주제')
    st.markdown('----')
    st.header('예시 데이터 다운로드')
    with open("dataset.zip", "rb") as z:
        btn = st.download_button(
            label="dataset.zip 다운로드",
            data=z,
            file_name="dataset.zip",
            mime="dataset/zip"
        )
    st.markdown('----')
    st.subheader('데이터 소개 및 진행 방법')
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    cl1 = col1.checkbox('EDA 사용법')
    if cl1 == True:
        st.subheader('EDA 종류')
        st.write('correlation은 이런걸 확인할 수 있음 등등')
    cl2 = col2.checkbox('방법론 1 사용법')
    if cl2 == True:
        st.subheader('방법론 1 사용법')
        st.write('방법론 1은 이렇게 사용하여 이런 결과를 볼 수 있음')
    cl3 = col3.checkbox('방법론 2 사용법')
    if cl3 == True:
        st.subheader('방법론 2 사용법')
        st.write('방법론 2는 이렇게 사용하여 이런 결과를 볼 수 있음')
    cl4 = col4.checkbox('결론 도출 방법')
    if cl4 == True:
        st.subheader('결론 도출 방법')
        st.write('이런과정으로 결론을 이렇게 도출할 수 있음')
    st.markdown('----')
    st.subheader('프로젝트 소개와 분석 방법, 이것저것 소개')
############################################################################

st.sidebar.header('데이터 불러오기')
df = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if df is not None:
    try:
        df = pd.read_csv(df)
    except:
        data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/dataset/train.csv"
        df = pd.read_csv(data_url)
else:
    data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/dataset/train.csv"
    df = pd.read_csv(data_url)

with st.sidebar:
    choose = option_menu("순서", ["1. 데이터 확인", "2. 변수 확인", "3. 결측치 확인", "4. error 변수 생성", "5. Total data 생성", '6. Type 변환'], icons=['bookmark-heart','bookmark-heart','bookmark-heart','bookmark-heart','bookmark-heart','bookmark-heart'],menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#D8D4C7"},
    }
    )
    
if introduce == False:
    st.subheader('1. 데이터 확인')
    data_check = st.checkbox('파일 데이터 확인')
    if data_check:
        st.write('로드된 데이터와 크기:', df.shape)
        st.write(df)
    df_col = df.columns

    st.markdown('----')

    st.subheader('2. 변수 확인')
    species_graph = st.checkbox('변수 확인 그래프')
    if species_graph:
        select_multi_species = st.selectbox('확인하고 싶은 변수 선택',df_col)
        if select_multi_species:
            st.subheader(select_multi_species + ' 그래프')
            select = df[select_multi_species]
            fig = px.bar(select, x=select.index, y=select.values)
            fig.update_layout(width=1100, height=500)
            st.plotly_chart(fig)         
        else:
            st.write('선택된 변수가 없습니다.')

    st.markdown('----')

    st.subheader('3. 결측치 확인')
    col5, col6 = st.columns([2,5])
    with col5:
        null_check = st.checkbox('결측지 여부 확인')
        if null_check:
            missing_values_count = df.isnull().sum()
            st.write('결측치 갯수는 : ', missing_values_count)
            with col6:
                fillna = st.checkbox('결측치 처리')
                st.markdown('★ 양품 --> 0 , 불양품 --> 1 ★')
                if fillna:
                    df = df.fillna(2)
                    df = df.replace('yes',0)
                    df = df.replace('no',1)
                    df = df
                    missing_values_count2 = df.isnull().sum()
                    st.write('결측치 갯수는 : ', missing_values_count2)
    
    st.markdown('----')
    st.subheader('4. error 변수 생성')
    st.markdown('★ machining_finalized 또는 passed_visual_inspection 하나라도 1이면 error ★')
    st.markdown("★ 'NO', 'material' 변수 삭제 ★")
    make_error = st.checkbox('error 생성')
    if make_error:
        for raw in range(len(df)):
            if df.loc[raw,'machining_finalized']==1:
                df.loc[raw,'error'] = 1
            elif df.loc[raw,'passed_visual_inspection']==1:
                df.loc[raw,'error'] = 1
            else:
                df.loc[raw,'error'] = 0
                
        df = df[['feedrate','clamp_pressure','tool_condition','error']]
        df['error'] = df['error'].astype('int')
        check = st.checkbox('데이터 확인')
        if check:
            st.write(df)

    st.markdown('----')
    st.subheader('5. Total data 생성')
    if st.checkbox('exp 데이터에 error 변수 추가'):
        try:
            file_path = './dataset/exp/'
            file = os.listdir(file_path)

            for i in range(len(file)):
                df1 = pd.read_csv(os.path.join(file_path,file[i]))
                #df1 = df1.drop(['Machining_Process'],axis=1)
                df1 = df1.drop(['M_sequence_number'],axis=1)
                df1 = df1.drop(['M_CURRENT_PROGRAM_NUMBER'],axis=1)
                df1['error']=df.loc[i,'error']
                try:
                    total_df = pd.concat([total_df,df1])
                    total_df = total_df.reset_index(drop=True)
                except:
                    total_df = df1
            st.write(total_df)
        except:
            st.write('error 변수를 만들어주세요')
    
    st.markdown('----')
    st.subheader('6. Type 변환')
    if st.checkbox('dtype 변환'):
        try:
            for col in total_df.columns:
                if total_df.dtypes[col] == "O":
                    uni = total_df[col].unique()
                    for i, val in enumerate(uni):
                        total_df[col] = total_df[col].replace(val, i)

            st.write('변환이 완료되었습니다.')
        except:
            total_df = total_df
            st.write('변환 가능한게 없습니다.')

    data_trans = st.button('데이터를 다른 페이지로 전송', help = '다른 페이지에서도 현제 페이지에서 설정한 데이터를 사용할 수 있습니다.')
    if data_trans == True:
        st.session_state['train_data'] = total_df
        st.write('데이터가 전송되었습니다.')
