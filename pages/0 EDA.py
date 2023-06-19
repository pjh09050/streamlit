import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import os
############################################################################
st.set_page_config(
    page_title="기말 프로젝트",
    page_icon="heart",
    layout="wide",
)
st.markdown(
    """
    <style>
    .main {
        max-width: 1400px;
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
#############################################################################
with st.sidebar:
    choose = option_menu("",["차례", "데이터 확인", "변수 확인", "결측치 확인", "error 변수 생성", "Total data 생성", 'Type 변환'], icons=['bar-chart','1-square', '2-square','3-square','4-square','5-square','6-square'],menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px", "background-color": "#D8D4C7"},
    }
    )

#############################################################################


if st.checkbox('데이터 불러오기'):
    data_url = "https://raw.githubusercontent.com/pjh09050/streamlit/main/dataset/train.csv"
    df = pd.read_csv(data_url)

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
                st.markdown('★ 양품 --> 0 , 불량품 --> 1 ★')
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
            file_path = 'dataset/exp/'
            file = os.listdir(file_path)
            a = 0
            for i in st.multiselect('합치고 싶은 데이터 선택', file, default=file):
                
                total = pd.read_csv(file_path + i)
                total = total.drop(['M_sequence_number'],axis=1)
                total = total.drop(['M_CURRENT_PROGRAM_NUMBER'],axis=1)
                total['error']=df.loc[a,'error']
                a += 1

                try:
                    total_df = pd.concat([total_df, total])
                    total_df = total_df.reset_index(drop=True)
                except:
                    total_df = total

            st.write(total_df.shape)
            st.write(total_df)

        except:
            st.write(':red[ error 변수를 만들어주세요]')

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
            st.write(total_df)
        except:
            total_df = total_df
            st.write('변환 가능한게 없습니다.')

    data_trans = st.button('데이터를 다른 페이지로 전송', help = '다른 페이지에서도 현제 페이지에서 설정한 데이터를 사용할 수 있습니다.')
    if data_trans == True:
        st.session_state['train_data'] = total_df
        st.write('데이터가 전송되었습니다.')
else:
    st.write('데이터를 불러와주세요')



