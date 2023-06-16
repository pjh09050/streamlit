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
import os

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
introduce = st.sidebar.checkbox('프로젝트 소개', value=False)
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
        df = pd.read_csv(df)
    except:
        data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/dataset/train.csv"
        df = pd.read_csv(data_url)
else:
    data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/dataset/train.csv"
    df = pd.read_csv(data_url)


# with st.sidebar:
#     choose = option_menu("데이터 확인", ["EDA", "방법론1", "방법론2"],icons=['bar-chart', 'bar-chart', 'bar-chart'],menu_icon="bi bi-card-list", default_index=0,
#         styles={
#         # default_index = 처음에 보여줄 페이지 인덱스 번호
#         "container": {"padding": "5!important", "background-color": "#fafafa"},
#         "icon": {"color": "black", "font-size": "20px"}, 
#         "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "#D8D4C7"},
#     } # css 설정
#     )
if introduce == False:
    st.subheader('1. 데이터 확인')
    data_check = st.checkbox('파일 데이터 확인')
    if data_check:
        st.write('로드된 데이터와 크기:', df.shape)
        st.write(df)
    df_col = df.columns

    st.subheader('2. 변수 확인')
    species_graph = st.checkbox('변수 확인 그래프 그리기')
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

    st.subheader('3. 결측치 확인')
    null = st.checkbox('결측치 확인하기')
    if null:
        missing_values_count = df.isnull().sum()
        st.write('결측치 갯수는 : ', missing_values_count)
        fillna = st.checkbox('결측치 처리하기')
        st.markdown('★ yes --> 0 , no --> 1 , 결측치 --> 2 ★')
        if fillna:
            df = df.fillna(2)
            df = df.replace('yes',0)
            df = df.replace('no',1)
            df = df
            missing_values_count2 = df.isnull().sum()
            st.write('결측치 갯수는 : ', missing_values_count2)

    st.subheader('4. error 변수 만들기')
    st.markdown('★ machining_finalized 또는 passed_visual_inspection 하나라도 1이면 error ★')
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

    st.subheader('5. 임시 제목 : Total data')
    if st.checkbox('exp 데이터에 error 변수 추가'):
        try:
            file_path = './dataset/exp/'
            file = os.listdir(file_path)

            for i in range(len(file)):
                df1 = pd.read_csv(os.path.join(file_path,file[i]))
                df1 = df1.drop(['Machining_Process'],axis=1)
                df1 = df1.drop(['M_sequence_number'],axis=1)
                df1 = df1.drop(['M_CURRENT_PROGRAM_NUMBER'],axis=1)
                #st.write(df)
                #st.write(df1)
                df1['error']=df.loc[i,'error']
                try:
                    total_df = pd.concat([total_df,df1])
                    total_df = total_df.reset_index(drop=True)
                except:
                    total_df = df1
            st.write(total_df)
        except:
            st.write('error 변수를 만들어주세요')

    data_trans = st.button('데이터를 다른 페이지로 전송', help = '다른 페이지에서도 현제 페이지에서 설정한 데이터를 사용할 수 있습니다.')
    if data_trans == True:
        st.session_state['train_data'] = total_df
        st.write('데이터가 전송되었습니다.')

    # st.subheader('4. Correlation')
    # select_heatmap = st.checkbox('Heatmap 그리기')
    # if select_heatmap:
    #     mask = np.zeros_like(df.corr(), dtype=np.bool_)
    #     mask[np.triu_indices_from(mask)] = True

    #     plt.figure(figsize=(10,4))
    #     sns.heatmap(data = df.corr(), annot=True, mask = mask,
    #                 fmt = '.2f', linewidths=.5, cmap='Blues')
    #     st.pyplot(plt.gcf())

        
# if choose == '방법론1' and introduce == False:
#     st.subheader('선택한 방법론 : 방법론1')
#     st.sidebar.header('파라미터 튜닝')
#     # 첫 번째 파라미터를 입력받는 텍스트 상자
#     param1 = st.sidebar.text_input('첫 번째 파라미터', '')
#     # 두 번째 파라미터를 입력받는 텍스트 상자
#     param2 = st.sidebar.text_input('두 번째 파라미터', '')
#     # 세 번째 파라미터를 입력받는 슬라이더
#     param3 = st.sidebar.slider('세 번째 파라미터', 0, 100, 50)

#     # 입력 받은 파라미터 값을 출력
#     st.write('첫 번째 파라미터:', param1)
#     st.write('두 번째 파라미터:', param2)
#     st.write('세 번째 파라미터:', param3)

# if choose == '방법론2' and introduce == False:
#     st.subheader('선택한 방법론 : 방법론2')
#     st.sidebar.header('파라미터 튜닝')
#     # 첫 번째 파라미터를 입력받는 텍스트 상자
#     param1 = st.sidebar.text_input('첫 번째 파라미터', '')
#     # 두 번째 파라미터를 입력받는 텍스트 상자
#     param2 = st.sidebar.text_input('두 번째 파라미터', '')
#     # 세 번째 파라미터를 입력받는 슬라이더
#     param3 = st.sidebar.slider('세 번째 파라미터', 0, 100, 50)

#     # 입력 받은 파라미터 값을 출력
#     st.write('첫 번째 파라미터 : ', param1)
#     st.write('두 번째 파라미터 : ', param2)
#     st.write('세 번째 파라미터 : ', param3)

