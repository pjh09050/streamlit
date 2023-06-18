import streamlit as st
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