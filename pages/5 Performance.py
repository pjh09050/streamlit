import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
import pandas as pd
device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn import metrics

st.subheader('Performance')

model_list = ['Logistic regression', 'Random Forest', 'DNN']
selected_models = []

col1, col2, col3 = st.columns([1,1,1])
cl1 = col1.checkbox('Logistic regression performance')
cl2 = col2.checkbox('Random Forest performance')
cl3 = col3.checkbox('DNN performance')

col4, col5, col6 = st.columns([1,1,1])

if cl1 == True:
    with st.spinner('Updating Report...'):
        try:
            y_pred_Logistic = st.session_state['y_pred_Logistic']
            y_test_Logistic = st.session_state['y_test_Logistic']
        except:
            st.write('Logistic 성능 데이터를 보내주세요')
            st.stop()

    accuracy_Logistic = metrics.accuracy_score(y_test_Logistic, y_pred_Logistic)
    precision_Logistic = metrics.precision_score(y_test_Logistic, y_pred_Logistic)
    recall_Logistic = metrics.recall_score(y_test_Logistic, y_pred_Logistic)
    f1_Logistic = metrics.f1_score(y_test_Logistic, y_pred_Logistic)
    data_Logistic = {'Accuracy' : [round(accuracy_Logistic, 4)],'Precision' : [round(precision_Logistic, 4)],'Recall' : [round(recall_Logistic, 4)],'F1-score' : [round(f1_Logistic, 4)]}
    df_Logistic = pd.DataFrame(data_Logistic, index=model_list[:1])
    selected_models.append(df_Logistic)


if cl2 == True:
    with st.spinner('Updating Report...'):
        try:
            y_pred_randomforest = st.session_state['y_pred_randomforest']
            y_test_randomforest = st.session_state['y_test_randomforest']
        except:
            st.write('randomforest 성능 데이터를 보내주세요')
            st.stop()

        accuracy_randomforest = metrics.accuracy_score(y_test_randomforest, y_pred_randomforest)
        precision_randomforest = metrics.precision_score(y_test_randomforest, y_pred_randomforest)
        recall_randomforest = metrics.recall_score(y_test_randomforest, y_pred_randomforest)
        f1_randomforest = metrics.f1_score(y_test_randomforest, y_pred_randomforest)
        data_randomforest = {'Accuracy' : [round(accuracy_randomforest, 4)],'Precision' : [round(precision_randomforest, 4)],'Recall' : [round(recall_randomforest, 4)],'F1-score' : [round(f1_randomforest, 4)],}
        df_randomforest = pd.DataFrame(data_randomforest, index=model_list[1:2])
        selected_models.append(df_randomforest)

if cl3 == True:
    with st.spinner('Updating Report...'):
        try:
            y_pred_dnn = st.session_state['y_pred_dnn']
            y_test_dnn = st.session_state['y_test_dnn']
        except:
            st.write('DNN 성능 데이터를 보내주세요')
            st.stop()

        accuracy_dnn = metrics.accuracy_score(y_test_dnn, y_pred_dnn)
        precision_dnn = metrics.precision_score(y_test_dnn, y_pred_dnn)
        recall_dnn = metrics.recall_score(y_test_dnn, y_pred_dnn)
        f1_dnn = metrics.f1_score(y_test_dnn, y_pred_dnn)
        data_DNN = {'Accuracy' : [round(accuracy_dnn, 4)],'Precision' : [round(precision_dnn, 4)],'Recall' : [round(recall_dnn, 4)],'F1-score' : [round(f1_dnn, 4)],}
        df_dnn = pd.DataFrame(data_DNN, index=model_list[2:])
        selected_models.append(df_dnn)


if selected_models:
    df_concat = pd.concat(selected_models)
    st.table(df_concat.style.set_properties(**{'text-align': 'left','font-size': '20px','width': '250px','height': '50px'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '2em')]}]))
elif cl1 or cl2 or cl3:
    st.write('선택한 모델에 대한 성능 데이터가 없습니다.')

