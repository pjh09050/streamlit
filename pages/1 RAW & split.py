import streamlit as st
from tensorflow.python.client import device_lib
from streamlit_option_menu import option_menu
import os
device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
############################################################################
with st.sidebar:
    choose = option_menu("", ["차례","데이터 확인", "변수 시각화", "데이터 셋 분할"], icons=['bar-chart','1-square','2-square','3-square'],menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px", "background-color": "#D8D4C7"},
    }
    )
############################################################################
with st.spinner('Updating Report...'):
    try:
        total_data = st.session_state['train_data']
    except:
        st.write(':red[ 데이터를 보내주세요]')
        st.stop()
############################################################################
if st.checkbox('데이터 확인'):
    st.write(total_data)
############################################################################
st.markdown('----')

st.subheader('RAW 데이터 시각화')

graph_list = ['선택', '그래프', '박스 플롯']
try:
    graph = st.selectbox('차트 선택', graph_list, index=0)
    if graph == '그래프':
        select_multi_species = st.selectbox('확인하고 싶은 변수 선택', total_data.columns)
        if select_multi_species:
            st.subheader(select_multi_species + ' 그래프')
            select = total_data[select_multi_species]
            fig = px.bar(select, x=select.index, y=select.values)
            fig.update_layout(width=1300, height=400)
            st.plotly_chart(fig)
        else:
            st.write('선택된 변수가 없습니다.')
############################################################################
    elif graph == '박스 플롯':
        select_box = st.selectbox('확인하고 싶은 변수 선택', total_data.columns)
        if select_box:
            st.subheader(select_box + ' 박스 플롯')
            fig = px.box(total_data, y=select_box, points="outliers", color='error')
            st.plotly_chart(fig)
        else:
            st.write('선택된 변수가 없습니다.')

############################################################################
    st.markdown('----')
    st.subheader('데이터 셋 분할')
    if st.checkbox('데이터 셋 분할'):
        ratio1 = st.number_input('학습 데이터 비율', min_value=0.0, max_value=1.0, value=0.8, step=0.1)
        ratio2 = st.number_input('테스트 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        if ratio1 + ratio2 != 1:
            st.write(':red[비율 합이 1이 되도록 설정해주세요.]')
        else:
            x_data = total_data.iloc[:,:-1].to_numpy()
            y_data = total_data.iloc[:,-1].to_numpy()
            
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio2, random_state=1)
            st.write('train 데이터 수 :', len(x_train))
            st.write('test 데이터 수 :', len(x_test))

            x_train = x_train
            x_test = x_test
            y_train = y_train
            y_test = y_test
            
            data_split = st.checkbox('데이터 분할')
            if data_split==True:
                st.write("데이터 분할을 완료했습니다. 학습을 시작해주세요.")
            data_pass = st.button('데이터를 다른 페이지로 전송')

            if data_pass == True:
                st.session_state['x_train_raw'] = x_train
                st.session_state['x_test_raw'] = x_test
                st.session_state['y_train_raw'] = y_train
                st.session_state['y_test_raw'] = y_test
                
                st.write('데이터가 전송되었습니다.')

except:
    st.markdown(':red[ 변수 선택해주세요]')