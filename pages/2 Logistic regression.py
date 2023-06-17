import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
device_lib.list_local_devices()
import plotly.express as px
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression

@st.cache_resource(ttl=24*60*60)
def import_logistic_predict_raw(x_train, x_test ,y_train ,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test
    
@st.cache_resource(ttl=24*60*60)
def import_logistic_predict_pca(x_train, x_test ,y_train ,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test
############################################################################
with st.sidebar:
    choose = option_menu("순서", ["데이터 선택", "데이터 학습", "성능 확인"], icons=['bookmark-heart','bookmark-heart','bookmark-heart'],menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#D8D4C7"},
    }
    )
############################################################################
st.subheader('Logistic regression')
col1, col2 = st.columns([1,5])
cl1 = col1.checkbox('RAW 데이터 사용')
cl2 = col2.checkbox('PCA 데이터 사용')
st.markdown('----')

if cl1 == True and cl2 == False:
    with st.spinner('Updating Report...'):
        try:
            x_train = st.session_state['x_train_raw']
            x_test = st.session_state['x_test_raw']
            y_train = st.session_state['y_train_raw']
            y_test = st.session_state['y_test_raw']
        except:
            st.write('데이터를 보내주세요')
            st.stop()
    st.subheader('데이터 학습')
    if st.checkbox('Logistic regression'):
        train_start = st.checkbox('RAW 데이터 학습 시작')
        if train_start == True:
            y_pred, y_test = import_logistic_predict_raw(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')

            if pred_check == True:
                classification_report_text = classification_report(y_test, y_pred)
                dnn_confusion_matrix = (confusion_matrix(y_test, y_pred))
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred)
                recall = metrics.recall_score(y_test, y_pred)
                f1 = metrics.f1_score(y_test, y_pred)

                col1, col2, col3 = st.columns([3,1,3])
                with col1:
                    st.code(classification_report_text)
                with col2:
                    st.write('Confusion Matrix:', dnn_confusion_matrix)
                with col3:
                    st.write('accuracy:', round(accuracy,4))
                    st.write('precision:', round(precision,4))
                    st.write('recall:', round(recall,4))
                    st.write('f1-score:', round(f1,4))

                col4, col5 = st.columns([1,1])
                fpr, tpr, dtc_thresholds = roc_curve(y_test, y_pred)
                fig_dtc_roc = px.area(x=fpr, y=tpr, title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=600, height=600)
                fig_dtc_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_dtc_roc.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_dtc_roc.update_xaxes(constrain='domain')
                with col4:
                    st.plotly_chart(fig_dtc_roc)
                with col5:
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('AUC: ', auc(fpr, tpr))
        
                st.markdown("모델이 종료되었습니다.")

                data_trans = st.button('성능 비교를 위한 데이터 전송')
                
                if data_trans == True:
                    st.session_state['y_pred_Logistic'] = y_pred
                    st.session_state['y_test_Logistic'] = y_test
                    st.write('Logistic 성능 데이터가 전송되었습니다.')


if cl1 == False and cl2 == True:
    with st.spinner('Updating Report...'):
        try:
            x_train = st.session_state['x_train']
            x_test = st.session_state['x_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
        except:
            st.write('데이터를 보내주세요')
            st.stop()

    st.subheader('데이터 학습')
    if st.checkbox('Logistic regression'):
        train_start = st.checkbox('PCA 데이터 학습 시작')
        if train_start == True:
            y_pred, y_test = import_logistic_predict_pca(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')

            if pred_check == True:
                classification_report_text = classification_report(y_test, y_pred)
                dnn_confusion_matrix = (confusion_matrix(y_test, y_pred))
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred)
                recall = metrics.recall_score(y_test, y_pred)
                f1 = metrics.f1_score(y_test, y_pred)

                col1, col2, col3 = st.columns([3,1,3])
                with col1:
                    st.code(classification_report_text)
                with col2:
                    st.write('Confusion Matrix:', dnn_confusion_matrix)
                with col3:
                    st.write('accuracy:', round(accuracy,4))
                    st.write('precision:', round(precision,4))
                    st.write('recall:', round(recall,4))
                    st.write('f1-score:', round(f1,4))

                col4, col5 = st.columns([1,1])
                fpr, tpr, dtc_thresholds = roc_curve(y_test, y_pred)
                fig_dtc_roc = px.area(x=fpr, y=tpr, title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=600, height=600)
                fig_dtc_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_dtc_roc.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_dtc_roc.update_xaxes(constrain='domain')
                with col4:
                    st.plotly_chart(fig_dtc_roc)
                with col5:
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('AUC: ', auc(fpr, tpr))
        
                st.markdown("모델이 종료되었습니다.")

                data_trans = st.button('성능 비교를 위한 데이터 전송')

                if data_trans == True:
                    st.session_state['y_pred_Logistic'] = y_pred
                    st.session_state['y_test_Logistic'] = y_test
                    st.write('Logistic 성능 데이터가 전송되었습니다.')