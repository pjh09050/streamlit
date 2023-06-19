import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
import pandas as pd
#device_lib.list_local_devices()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn import metrics
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
        max-width: 1500px;
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
    choose = option_menu("",  ['Models','Logistic Regression', 'Random Forest', 'DNN', 'XGBOOST', 'SVM'], icons=['reception-4', 'battery-full','battery-full','battery-full','battery-full','battery-full'], menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px","background-color": "#D8D4C7"},
    }
    )
    choose1 = option_menu("",  ['Performance','Accuracy', 'Precision', 'Recall', 'F1-Score'], icons=['reception-4', 'battery-full','battery-full','battery-full','battery-full'], menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px","background-color": "#D8D4C7"},
    }
    )
############################################################################
st.subheader('Performance')
model_list = ['Logistic Regression_raw','Logistic Regression_PCA', 'Random Forest_raw','Random Forest_PCA', 'DNN_raw', 'DNN_PCA', 'XGB_raw' ,'XGB_PCA', 'SVM_raw', 'SVM_PCA']
performance_select = st.multiselect('모델 선택', model_list)
selected_models = []
############################################################################
if 'Logistic Regression_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_Logistic_raw = st.session_state['y_pred_Logistic_raw']
            y_test_Logistic_raw = st.session_state['y_test_Logistic_raw']
        except:
            st.write(':red[ Logistic_raw 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_Logistic_raw = metrics.accuracy_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    precision_Logistic_raw = metrics.precision_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    recall_Logistic_raw = metrics.recall_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    f1_Logistic_raw = metrics.f1_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    data_Logistic_raw = {'Accuracy' : [round(accuracy_Logistic_raw, 4)],'Precision' : [round(precision_Logistic_raw, 4)],'Recall' : [round(recall_Logistic_raw, 4)],'F1-score' : [round(f1_Logistic_raw, 4)],}
    df_Logistic_raw = pd.DataFrame(data_Logistic_raw, index=model_list[:1])
    selected_models.append(df_Logistic_raw)

if 'Logistic Regression_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_Logistic_pca = st.session_state['y_pred_Logistic_PCA']
            y_test_Logistic_pca = st.session_state['y_test_Logistic_PCA']
        except:
            st.write(':red[ Logistic_pca 성능 데이터를 보내주세요]')
    accuracy_Logistic_pca = metrics.accuracy_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    precision_Logistic_pca = metrics.precision_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    recall_Logistic_pca = metrics.recall_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    f1_Logistic_pca = metrics.f1_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    data_Logistic_pca = {'Accuracy' : [round(accuracy_Logistic_pca, 4)],'Precision' : [round(precision_Logistic_pca, 4)],'Recall' : [round(recall_Logistic_pca, 4)],'F1-score' : [round(f1_Logistic_pca, 4)],}
    df_Logistic_pca = pd.DataFrame(data_Logistic_pca, index=model_list[1:2])
    selected_models.append(df_Logistic_pca)

############################################################################
if 'Random Forest_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_randomforest_raw = st.session_state['y_pred_randomforest_raw']
            y_test_randomforest_raw = st.session_state['y_test_randomforest_raw']
        except:
            st.write(':red[ randomforest_raw 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_randomforest_raw = metrics.accuracy_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    precision_randomforest_raw = metrics.precision_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    recall_randomforest_raw = metrics.recall_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    f1_randomforest_raw = metrics.f1_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    data_randomforest_raw = {'Accuracy' : [round(accuracy_randomforest_raw, 4)],'Precision' : [round(precision_randomforest_raw, 4)],'Recall' : [round(recall_randomforest_raw, 4)],'F1-score' : [round(f1_randomforest_raw, 4)],}
    df_randomforest_raw = pd.DataFrame(data_randomforest_raw, index=model_list[2:3])
    selected_models.append(df_randomforest_raw)

if 'Random Forest_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_randomforest_pca = st.session_state['y_pred_randomforest_PCA']
            y_test_randomforest_pca = st.session_state['y_test_randomforest_PCA']
        except:
            st.write(':red[ randomforest_pca 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_randomforest_pca = metrics.accuracy_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    precision_randomforest_pca = metrics.precision_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    recall_randomforest_pca = metrics.recall_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    f1_randomforest_pca = metrics.f1_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    data_randomforest_pca = {'Accuracy' : [round(accuracy_randomforest_pca, 4)],'Precision' : [round(precision_randomforest_pca, 4)],'Recall' : [round(recall_randomforest_pca, 4)],'F1-score' : [round(f1_randomforest_pca, 4)],}
    df_randomforest_pca = pd.DataFrame(data_randomforest_pca, index=model_list[3:4])
    selected_models.append(df_randomforest_pca)

############################################################################
if 'DNN_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_dnn_raw = st.session_state['y_pred_dnn_raw']
            y_test_dnn_raw = st.session_state['y_test_dnn_raw']
        except:
            st.write(':red[ DNN_raw 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_dnn_raw = metrics.accuracy_score(y_test_dnn_raw, y_pred_dnn_raw)
    precision_dnn_raw = metrics.precision_score(y_test_dnn_raw, y_pred_dnn_raw)
    recall_dnn_raw = metrics.recall_score(y_test_dnn_raw, y_pred_dnn_raw)
    f1_dnn_raw = metrics.f1_score(y_test_dnn_raw, y_pred_dnn_raw)
    data_DNN_raw = {'Accuracy' : [round(accuracy_dnn_raw, 4)],'Precision' : [round(precision_dnn_raw, 4)],'Recall' : [round(recall_dnn_raw, 4)],'F1-score' : [round(f1_dnn_raw, 4)],}
    df_dnn_raw = pd.DataFrame(data_DNN_raw, index=model_list[4:5])
    selected_models.append(df_dnn_raw)

if 'DNN_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_dnn_pca = st.session_state['y_pred_dnn_PCA']
            y_test_dnn_pca = st.session_state['y_test_dnn_PCA']
        except:
            st.write(':red[ DNN_pca 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_dnn_pca = metrics.accuracy_score(y_test_dnn_pca, y_pred_dnn_pca)
    precision_dnn_pca = metrics.precision_score(y_test_dnn_pca, y_pred_dnn_pca)
    recall_dnn_pca = metrics.recall_score(y_test_dnn_pca, y_pred_dnn_pca)
    f1_dnn_pca = metrics.f1_score(y_test_dnn_pca, y_pred_dnn_pca)
    data_DNN_pca = {'Accuracy' : [round(accuracy_dnn_pca, 4)],'Precision' : [round(precision_dnn_pca, 4)],'Recall' : [round(recall_dnn_pca, 4)],'F1-score' : [round(f1_dnn_pca, 4)],}
    df_dnn_pca = pd.DataFrame(data_DNN_pca, index=model_list[5:6])
    selected_models.append(df_dnn_pca)

############################################################################
if 'XGB_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_XGB_raw = st.session_state['y_pred_XGB_raw']
            y_test_XGB_raw = st.session_state['y_test_XGB_raw']
        except:
            st.write(':red[ XGB_raw 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_XGB_raw = metrics.accuracy_score(y_test_XGB_raw, y_pred_XGB_raw)
    precision_XGB_raw = metrics.precision_score(y_test_XGB_raw, y_pred_XGB_raw)
    recall_XGB_raw = metrics.recall_score(y_test_XGB_raw, y_pred_XGB_raw)
    f1_XGB_raw = metrics.f1_score(y_test_XGB_raw, y_pred_XGB_raw)
    data_XGB_raw = {'Accuracy' : [round(accuracy_XGB_raw, 4)],'Precision' : [round(precision_XGB_raw, 4)],'Recall' : [round(recall_XGB_raw, 4)],'F1-score' : [round(f1_XGB_raw, 4)],}
    df_XGB_raw = pd.DataFrame(data_XGB_raw, index=model_list[6:7])
    selected_models.append(df_XGB_raw)

if 'XGB_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_XGB_pca = st.session_state['y_pred_XGB_PCA']
            y_test_XGB_pca = st.session_state['y_test_XGB_PCA']
        except:
            st.write(':red[ XGB_pca 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_XGB_pca = metrics.accuracy_score(y_test_XGB_pca, y_pred_XGB_pca)
    precision_XGB_pca = metrics.precision_score(y_test_XGB_pca, y_pred_XGB_pca)
    recall_XGB_pca = metrics.recall_score(y_test_XGB_pca, y_pred_XGB_pca)
    f1_XGB_pca = metrics.f1_score(y_test_XGB_pca, y_pred_XGB_pca)
    data_XGB_pca = {'Accuracy' : [round(accuracy_XGB_pca, 4)],'Precision' : [round(precision_XGB_pca, 4)],'Recall' : [round(recall_XGB_pca, 4)],'F1-score' : [round(f1_XGB_pca, 4)],}
    df_XGB_pca = pd.DataFrame(data_XGB_pca, index=model_list[7:8])
    selected_models.append(df_XGB_pca)

############################################################################
if 'SVM_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_SVM_raw = st.session_state['y_pred_SVM_raw']
            y_test_SVM_raw = st.session_state['y_test_SVM_raw']
        except:
            st.write(':red[ SVM_raw 성능 데이터를 보내주세요]')
            st.stop()
    accuracy_SVM_raw = metrics.accuracy_score(y_test_SVM_raw, y_pred_SVM_raw)
    precision_SVM_raw = metrics.precision_score(y_test_SVM_raw, y_pred_SVM_raw)
    recall_SVM_raw = metrics.recall_score(y_test_SVM_raw, y_pred_SVM_raw)
    f1_SVM_raw = metrics.f1_score(y_test_SVM_raw, y_pred_SVM_raw)
    data_SVM_raw = {'Accuracy' : [round(accuracy_SVM_raw, 4)],'Precision' : [round(precision_SVM_raw, 4)],'Recall' : [round(recall_SVM_raw, 4)],'F1-score' : [round(f1_SVM_raw, 4)],}
    df_SVM_raw = pd.DataFrame(data_SVM_raw, index=model_list[8:9])
    selected_models.append(df_SVM_raw)

if 'SVM_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            y_pred_SVM_pca = st.session_state['y_pred_SVM_PCA']
            y_test_SVM_pca = st.session_state['y_test_SVM_PCA']
        except:
            st.write(':red[ SVM_pca 성능 데이터를 보내주세요]')
    accuracy_SVM_pca = metrics.accuracy_score(y_test_SVM_pca, y_pred_SVM_pca)
    precision_SVM_pca = metrics.precision_score(y_test_SVM_pca, y_pred_SVM_pca)
    recall_SVM_pca = metrics.recall_score(y_test_SVM_pca, y_pred_SVM_pca)
    f1_SVM_pca = metrics.f1_score(y_test_SVM_pca, y_pred_SVM_pca)
    data_SVM_pca = {'Accuracy' : [round(accuracy_SVM_pca, 4)],'Precision' : [round(precision_SVM_pca, 4)],'Recall' : [round(recall_SVM_pca, 4)],'F1-score' : [round(f1_SVM_pca, 4)],}
    df_SVM_pca = pd.DataFrame(data_SVM_pca, index=model_list[9:10])
    selected_models.append(df_SVM_pca)

############################################################################
if selected_models:
    df_concat = pd.concat(selected_models)
    st.table(df_concat.style.set_properties(**{'text-align': 'left','font-size': '18px','width': '250px','height': '50px'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '1.5em')]}]))

############################################################################
