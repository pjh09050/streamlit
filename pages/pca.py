import streamlit as st

with st.spinner('Updating Report...'):
    df = st.session_state['x_train'] 
    st.write('데이터가 전송되었습니다.')