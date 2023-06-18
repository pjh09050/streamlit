import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
import pandas as pd
device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn import metrics



############################################################################
with st.sidebar:
    choose = option_menu("Model", ['Logistic Regression', 'Random Forest', 'Deep Neural Network'], icons=['1-square','2-square','3-square'], menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "18px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#D8D4C7"},
    }
    )
############################################################################
st.subheader('Performance')
model_list = ['Logistic Regression_raw','Logistic Regression_PCA', 'Random Forest_raw','Random Forest_PCA', 'DNN_raw', 'DNN_PCA']
performance_select = st.multiselect('모델 성능 확인 : ', model_list)
selected_models = []
############################################################################
with st.spinner('Updating Report...'):
    try:
        y_pred_Logistic_raw = st.session_state['y_pred_Logistic_raw']
        y_test_Logistic_raw = st.session_state['y_test_Logistic_raw']
    except:
        st.write('Logistic_raw 성능 데이터를 보내주세요')
        st.stop()

with st.spinner('Updating Report...'):
    try:
        y_pred_Logistic_pca = st.session_state['y_pred_Logistic_PCA']
        y_test_Logistic_pca = st.session_state['y_test_Logistic_PCA']
    except:
        st.write('Logistic_pca 성능 데이터를 보내주세요')
############################################################################
with st.spinner('Updating Report...'):
    try:
        y_pred_randomforest_raw = st.session_state['y_pred_randomforest_raw']
        y_test_randomforest_raw = st.session_state['y_test_randomforest_raw']
    except:
        st.write('randomforest_raw 성능 데이터를 보내주세요')
        st.stop()

with st.spinner('Updating Report...'):
    try:
        y_pred_randomforest_pca = st.session_state['y_pred_randomforest_PCA']
        y_test_randomforest_pca = st.session_state['y_test_randomforest_PCA']
    except:
        st.write('randomforest_pca 성능 데이터를 보내주세요')
        st.stop()
############################################################################
with st.spinner('Updating Report...'):
    try:
        y_pred_dnn_raw = st.session_state['y_pred_dnn_raw']
        y_test_dnn_raw = st.session_state['y_test_dnn_raw']
    except:
        st.write('DNN_raw 성능 데이터를 보내주세요')
        st.stop()

with st.spinner('Updating Report...'):
    try:
        y_pred_dnn_pca = st.session_state['y_pred_dnn_PCA']
        y_test_dnn_pca = st.session_state['y_test_dnn_PCA']
    except:
        st.write('DNN_pca 성능 데이터를 보내주세요')
        st.stop()
############################################################################

if 'Logistic Regression_raw' in performance_select:
    accuracy_Logistic_raw = metrics.accuracy_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    precision_Logistic_raw = metrics.precision_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    recall_Logistic_raw = metrics.recall_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    f1_Logistic_raw = metrics.f1_score(y_test_Logistic_raw, y_pred_Logistic_raw)
    data_Logistic_raw = {'Accuracy' : [round(accuracy_Logistic_raw, 4)],'Precision' : [round(precision_Logistic_raw, 4)],'Recall' : [round(recall_Logistic_raw, 4)],'F1-score' : [round(f1_Logistic_raw, 4)],}
    df_Logistic_raw = pd.DataFrame(data_Logistic_raw, index=model_list[:1])
    selected_models.append(df_Logistic_raw)

if 'Logistic Regression_PCA' in performance_select:
    accuracy_Logistic_pca = metrics.accuracy_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    precision_Logistic_pca = metrics.precision_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    recall_Logistic_pca = metrics.recall_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    f1_Logistic_pca = metrics.f1_score(y_test_Logistic_pca, y_pred_Logistic_pca)
    data_Logistic_pca = {'Accuracy' : [round(accuracy_Logistic_pca, 4)],'Precision' : [round(precision_Logistic_pca, 4)],'Recall' : [round(recall_Logistic_pca, 4)],'F1-score' : [round(f1_Logistic_pca, 4)],}
    df_Logistic_pca = pd.DataFrame(data_Logistic_pca, index=model_list[1:2])
    selected_models.append(df_Logistic_pca)

############################################################################
if 'Random Forest_raw' in performance_select:
    accuracy_randomforest_raw = metrics.accuracy_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    precision_randomforest_raw = metrics.precision_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    recall_randomforest_raw = metrics.recall_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    f1_randomforest_raw = metrics.f1_score(y_test_randomforest_raw, y_pred_randomforest_raw)
    data_randomforest_raw = {'Accuracy' : [round(accuracy_randomforest_raw, 4)],'Precision' : [round(precision_randomforest_raw, 4)],'Recall' : [round(recall_randomforest_raw, 4)],'F1-score' : [round(f1_randomforest_raw, 4)],}
    df_randomforest_raw = pd.DataFrame(data_randomforest_raw, index=model_list[2:3])
    selected_models.append(df_randomforest_raw)

if 'Random Forest_PCA' in performance_select:
    accuracy_randomforest_pca = metrics.accuracy_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    precision_randomforest_pca = metrics.precision_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    recall_randomforest_pca = metrics.recall_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    f1_randomforest_pca = metrics.f1_score(y_test_randomforest_pca, y_pred_randomforest_pca)
    data_randomforest_pca = {'Accuracy' : [round(accuracy_randomforest_pca, 4)],'Precision' : [round(precision_randomforest_pca, 4)],'Recall' : [round(recall_randomforest_pca, 4)],'F1-score' : [round(f1_randomforest_pca, 4)],}
    df_randomforest_pca = pd.DataFrame(data_randomforest_pca, index=model_list[3:4])
    selected_models.append(df_randomforest_pca)

############################################################################
if 'DNN_raw' in performance_select:
    accuracy_dnn_raw = metrics.accuracy_score(y_test_dnn_raw, y_pred_dnn_raw)
    precision_dnn_raw = metrics.precision_score(y_test_dnn_raw, y_pred_dnn_raw)
    recall_dnn_raw = metrics.recall_score(y_test_dnn_raw, y_pred_dnn_raw)
    f1_dnn_raw = metrics.f1_score(y_test_dnn_raw, y_pred_dnn_raw)
    data_DNN_raw = {'Accuracy' : [round(accuracy_dnn_raw, 4)],'Precision' : [round(precision_dnn_raw, 4)],'Recall' : [round(recall_dnn_raw, 4)],'F1-score' : [round(f1_dnn_raw, 4)],}
    df_dnn_raw = pd.DataFrame(data_DNN_raw, index=model_list[4:5])
    selected_models.append(df_dnn_raw)

if 'DNN_PCA' in performance_select:
    accuracy_dnn_pca = metrics.accuracy_score(y_test_dnn_pca, y_pred_dnn_pca)
    precision_dnn_pca = metrics.precision_score(y_test_dnn_pca, y_pred_dnn_pca)
    recall_dnn_pca = metrics.recall_score(y_test_dnn_pca, y_pred_dnn_pca)
    f1_dnn_pca = metrics.f1_score(y_test_dnn_pca, y_pred_dnn_pca)
    data_DNN_pca = {'Accuracy' : [round(accuracy_dnn_pca, 4)],'Precision' : [round(precision_dnn_pca, 4)],'Recall' : [round(recall_dnn_pca, 4)],'F1-score' : [round(f1_dnn_pca, 4)],}
    df_dnn_pca = pd.DataFrame(data_DNN_pca, index=model_list[5:6])
    selected_models.append(df_dnn_pca)

############################################################################
if selected_models:
    df_concat = pd.concat(selected_models)
    st.table(df_concat.style.set_properties(**{'text-align': 'left','font-size': '20px','width': '250px','height': '50px'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '2em')]}]))

############################################################################
