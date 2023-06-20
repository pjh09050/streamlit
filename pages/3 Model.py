import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
import numpy as np
#device_lib.list_local_devices()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import plotly.express as px
from sklearn import metrics
from tool.model.DL_model import dnn
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import sklearn.svm as svm
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
@st.cache_resource(ttl=24*60*60)
def import_LR_train(x_train, x_test ,y_train ,y_test):
    LR = LogisticRegression()
    LR.fit(x_train, y_train)

    return LR

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_SVM_train(x_train, x_test ,y_train ,y_test):
    
    SVM = svm.SVC(kernel='rbf', C=3, gamma=0.25, probability=True)
    SVM.fit(x_train, y_train)
    
    return SVM

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_RF_train(x_train, x_test ,y_train ,y_test):
    RF = RandomForestClassifier(n_estimators = 200, criterion='entropy', bootstrap=True, random_state=42, max_depth=10,min_samples_leaf=5,min_samples_split=2)
    RF.fit(x_train, y_train)

    return RF

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_XGB_train(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    XGB = xgb.XGBClassifier(objective='binary:logistic',
                            min_child_weight=1,
                            reg_lambda=1,
                            num_boost_round = 500,
                            colsample_bytree=0.8,
                            reg_alpha=0.1,
                            n_estimators=400,
                            learning_rate=0.1,
                            max_depth=10)
    XGB.fit(x_train,y_train,
            early_stopping_rounds=100,
            eval_metric='error',
            eval_set = [(x_val,y_val)],
            verbose=True)
    return XGB

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_DNN_train(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    
    DNN = dnn(x_train.shape[1], len(np.unique(y_test))-1)
    DNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=6, mode='min')]
    
    DNN.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), batch_size=128, callbacks=callbacks, verbose=0)
    
    return DNN

############################################################################
def import_evaluate(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
    except:
        if model.model_name == 'DNN':
            x_test = x_test.reshape(x_test.shape[0],-1,1)
            y_pred = model.predict(x_test)
            def predict_proba(model, x, batch_size=32, verbose=1):
                preds = model.predict(x, batch_size, verbose)
                return preds
            y_pred_proba = predict_proba(model, x_test)

    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
            if y_pred[y] >= 0.5:
                y_pred[y] = 1
            else:
                y_pred[y] = 0
    
    classification_report_text = classification_report(y_test, y_pred)
    cm = (confusion_matrix(y_test, y_pred))
    accuracy = round(metrics.accuracy_score(y_test, y_pred),4)
    precision = round(metrics.precision_score(y_test, y_pred),4)
    recall = round(metrics.recall_score(y_test, y_pred),4)
    f1 = round(metrics.f1_score(y_test, y_pred),4)
    
    fpr, tpr, dtc_thresholds = roc_curve(y_test, y_pred_proba)
    fig_dtc_roc = px.area(x=fpr, y=tpr, title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=600, height=600)
    fig_dtc_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig_dtc_roc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_dtc_roc.update_xaxes(constrain='domain')
    
    AUC = round(auc(fpr, tpr),4)
    
    return classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC

############################################################################
with st.sidebar:
<<<<<<< HEAD
    choose = option_menu("", ['Models','로지스틱 회귀', '서포트벡터머신', '랜덤 포레스트', 'XGBOOST', 'Deep Neural Network'], icons=['reception-3', 'battery-full','battery-full','battery-full','battery-full','battery-full'], menu_icon="bi bi-card-list",
=======
    choose = option_menu("", ['Models','Logistic Regression', 'Random Forest', 'DNN', 'XGBOOST', 'SVM'], icons=['reception-3', 'battery-full','battery-full','battery-full','battery-full','battery-full'], menu_icon="bi bi-card-list",
>>>>>>> f8b2d8b54920de8b222111d96d45b7a43014439a
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px","background-color": "#D8D4C7"},
    }
    )
############################################################################

st.subheader('Model')

model_select = st.selectbox("모델 선택", ['선택', '로지스틱 회귀', '서포트벡터머신', '랜덤 포레스트', 'XGBOOST', 'Deep Neural Network'], index=0)

if model_select == '로지스틱 회귀':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()
        st.subheader('데이터 학습')
        if st.checkbox('로지스틱 회귀'):
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                LR_raw = import_LR_train(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_LR_raw = st.button('성능 비교를 위한 데이터 전송')                       
                if data_trans_LR_raw == True:
                    st.write('로지스틱 회귀 성능 데이터가 전송되었습니다.')

                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(LR_raw, x_test, y_test)

                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)

                    if data_trans_LR_raw == True:
                        LR_raw_metric = {'로지스틱 회귀_raw':[accuracy, precision, recall, f1, AUC]}
                        st.session_state['LR_raw_metric'] = LR_raw_metric
                    
                    model_LR_raw = st.button('예측을 위한 모델 전송')
                    if model_LR_raw == True:
                        st.session_state['LR_raw'] = LR_raw
                        st.markdown("모델이 전송되었습니다.")
                        
############################################################################
                        
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('로지스틱 회귀'):
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                LR_pca = import_LR_train(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_LR_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_LR_pca == True:
                    st.write('로지스틱 회귀 성능 데이터가 전송되었습니다.')
                    
                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(LR_pca, x_test, y_test)

                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)

                    if data_trans_LR_pca == True:
                        LR_pca_metric = {'로지스틱 회귀_pca' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['LR_pca_metric'] = LR_pca_metric
                        
                    model_LR_pca = st.button('예측을 위한 모델 전송')
                    if model_LR_pca == True:
                        st.session_state['LR_pca'] = LR_pca
                        st.markdown("모델이 전송되었습니다.")

############################################################################
if model_select == '서포트벡터머신':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()
        st.subheader('데이터 학습')
        if st.checkbox('서포트벡터머신'):
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                SVM_raw = import_SVM_train(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_SVM_raw = st.button('성능 비교를 위한 데이터 전송')                       
                if data_trans_SVM_raw == True:
                    st.write('서포트벡터머신 성능 데이터가 전송되었습니다.')

                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(SVM_raw, x_test, y_test)
                    
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)

                    if data_trans_SVM_raw == True:
                        SVM_raw_metric = {'서포트벡터머신_raw' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['SVM_raw_metric'] = SVM_raw_metric
                        
                    model_SVM_raw = st.button('예측을 위한 모델 전송')
                    if model_SVM_raw == True:
                        st.session_state['SVM_raw'] = SVM_raw
                        st.markdown("모델이 전송되었습니다.")
                        
############################################################################
                        
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('서포트벡터머신'):
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                SVM_pca = import_SVM_train(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_SVM_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_SVM_pca == True:
                    st.write('서포트벡터머신 성능 데이터가 전송되었습니다.')
                    
                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(SVM_pca, x_test, y_test)
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
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
                        st.write('AUC: ', AUC)

                    if data_trans_SVM_pca == True:
                        SVM_pca_metric = {'서포트벡터머신_pca' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['SVM_pca_metric'] = SVM_pca_metric
                        
                    model_SVM_pca = st.button('예측을 위한 모델 전송')
                    if model_SVM_pca == True:
                        st.session_state['SVM_pca'] = SVM_pca
                        st.markdown("모델이 전송되었습니다.")

############################################################################    
<<<<<<< HEAD
if model_select == '랜덤 포레스트':
=======
if model_select == 'Random Forest':
>>>>>>> f8b2d8b54920de8b222111d96d45b7a43014439a
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()
        st.subheader('데이터 학습')
        train_start = st.checkbox('RAW 데이터 학습 시작')
        if train_start == True:
            RF_raw = import_RF_train(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')
            data_trans_RF_raw = st.button('성능 비교를 위한 데이터 전송')
            if data_trans_RF_raw == True:
                st.write('랜덤 포레스트 성능 데이터가 전송되었습니다.')
            if pred_check == True:
                classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(RF_raw, x_test, y_test)
                col1, col2, col3 = st.columns([3,1,3])
                with col1:
                    st.code(classification_report_text)
                with col2:
                    st.write('Confusion Matrix:', cm)
                with col3:
                    st.write('정확도:', round(accuracy,4))
                    st.write('정밀도:', round(precision,4))
                    st.write('재현율:', round(recall,4))
                    st.write('f1-score:', round(f1,4))

                col4, col5 = st.columns([1,1])
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
                    st.write('AUC: ', AUC)

                if data_trans_RF_raw == True:
                    RF_raw_metric = {'랜덤 포레스트_raw' : [accuracy, precision, recall, f1, AUC]}
                    st.session_state['RF_raw_metric'] = RF_raw_metric

                model_RF_raw = st.button('예측을 위한 모델 전송')
                if model_RF_raw == True:
                    st.session_state['RF_raw'] = RF_raw
                    st.markdown("모델이 전송되었습니다.")
                
############################################################################ 
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        train_start = st.checkbox('PCA 데이터 학습 시작')
        if train_start == True:
            RF_pca = import_RF_train(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')
            data_trans_RF_pca = st.button('성능 비교를 위한 데이터 전송')
            if data_trans_RF_pca == True:
                st.write('랜덤 포레스트 성능 데이터가 전송되었습니다.')

            if pred_check == True:
                classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(RF_pca, x_test, y_test)
                col1, col2, col3 = st.columns([3,1,3])
                with col1:
                    st.code(classification_report_text)
                with col2:
                    st.write('Confusion Matrix:', cm)
                with col3:
                    st.write('정확도:', round(accuracy,4))
                    st.write('정밀도:', round(precision,4))
                    st.write('재현율:', round(recall,4))
                    st.write('f1-score:', round(f1,4)), round(f1,4)
                
                col4, col5 = st.columns([1,1])
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
                    st.write('AUC: ', AUC)

                if data_trans_RF_pca == True:
                    RF_pca_metric = {'랜덤 포레스트_pca' : [accuracy, precision, recall, f1, AUC]}
                    st.session_state['RF_pca_metric'] = RF_pca_metric

                model_RF_pca = st.button('예측을 위한 모델 전송')
                if model_RF_pca == True:
                    st.session_state['RF_pca'] = RF_pca
                    st.markdown("모델이 전송되었습니다.")
                
############################################################################   
if model_select == 'XGBOOST':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('데이터 학습'):
            
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]
            
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                XGB_raw = import_XGB_train(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_XGB_raw = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_XGB_raw == True:
                    st.write('XGBOOST 성능 데이터가 전송되었습니다.')
                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(XGB_raw, x_test, y_test)
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)

                    if data_trans_XGB_raw == True:
                        XGB_raw_metric = {'XGBOOST_raw' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['XGB_raw_metric'] = XGB_raw_metric

                    model_XGB_raw = st.button('예측을 위한 모델 전송')
                    if model_XGB_raw == True:
                        st.session_state['XGB_raw'] = XGB_raw
                        st.markdown("모델이 전송되었습니다.")
                
############################################################################ 
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('데이터 학습'):
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]
            
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                XGB_pca = import_XGB_train(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_XGB_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_XGB_pca == True:
                    st.write('XGBOOST 성능 데이터가 전송되었습니다.')

                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(XGB_pca, x_test, y_test)
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)

                    if data_trans_XGB_pca == True:
                        XGB_pca_metric = {'XGBOOST_pca' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['XGB_pca_metric'] = XGB_pca_metric

                    model_XGB_pca = st.button('예측을 위한 모델 전송')
                    if model_XGB_pca == True:
                        st.session_state['XGB_pca'] = XGB_pca
                        st.markdown("모델이 전송되었습니다.")


############################################################################         
if model_select == 'Deep Neural Network':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('데이터 학습'):
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]
            x_train = x_train.reshape(x_train.shape[0],-1,1)
            x_val = x_val.reshape(x_val.shape[0],-1,1)
            x_test = x_test.reshape(x_test.shape[0],-1,1)

            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                DNN_raw = import_DNN_train(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_DNN_raw = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_DNN_raw == True:
                    st.write('DNN 성능 데이터가 전송되었습니다.')

                if pred_check == True:
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(DNN_raw, x_test, y_test)
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)
            
                    st.markdown("모델이 종료되었습니다.")

                    if data_trans_DNN_raw == True:
                        DNN_raw_metric = {'DNN_raw' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['DNN_raw_metric'] = DNN_raw_metric

                    model_DNN_raw = st.button('예측을 위한 모델 전송')
                    if model_DNN_raw == True:
                        st.session_state['DNN_raw'] = DNN_raw
                        st.markdown("모델이 전송되었습니다.")
                        
############################################################################  
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('PCA 데이터 학습 시작'):
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]

            x_train = x_train.reshape(x_train.shape[0],-1,1)
            x_val = x_val.reshape(x_val.shape[0],-1,1)
            x_test = x_test.reshape(x_test.shape[0],-1,1)

            train_start = st.checkbox('DNN 학습 시작')
            if train_start == True:
                DNN_pca = import_DNN_train(x_train, x_val, x_test ,y_train ,y_val ,y_test)

                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인하기')
                data_trans_DNN_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_DNN_pca == True:
                    st.write('DNN 성능 데이터가 전송되었습니다.')

                if pred_check == True:
<<<<<<< HEAD
                    classification_report_text, cm, accuracy, precision, recall, f1, fig_dtc_roc, AUC = import_evaluate(DNN_pca, x_test, y_test)
=======
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
            
                    if data_trans_dnn_pca == True:
                        st.session_state['y_pred_dnn_PCA'] = y_pred
                        st.session_state['y_test_dnn_PCA'] = y_test                                  
                    st.markdown("모델이 종료되었습니다.")

############################################################################   
if model_select == 'XGBOOST':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('데이터 학습'):
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]
            
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test = import_XGB_predict_raw(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_XGB_raw = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_XGB_raw == True:
                    st.write('XGBOOST 성능 데이터가 전송되었습니다.')
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

                    if data_trans_XGB_raw == True:
                        st.session_state['y_pred_XGB_raw'] = y_pred
                        st.session_state['y_test_XGB_raw'] = y_test            
                    st.markdown("모델이 종료되었습니다.")
                
############################################################################ 
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('데이터 학습'):
            ratio1 = st.number_input('검증 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            train_len = len(x_train)
            val_len = int(train_len * ratio1)

            x_val = x_train[:val_len]
            y_val = y_train[:val_len]

            x_train = x_train[val_len:]
            y_train = y_train[val_len:]
            
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test = import_XGB_predict_pca(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_XGB_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_XGB_pca == True:
                    st.write('XGBOOST 성능 데이터가 전송되었습니다.')

                if pred_check == True:
                    classification_report_text = classification_report(y_test, y_pred)
                    XGB_confusion_matrix = (confusion_matrix(y_test, y_pred))
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                    precision = metrics.precision_score(y_test, y_pred)
                    recall = metrics.recall_score(y_test, y_pred)
                    f1 = metrics.f1_score(y_test, y_pred)

                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', XGB_confusion_matrix)
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

                    if data_trans_XGB_pca == True:
                        st.session_state['y_pred_XGB_PCA'] = y_pred
                        st.session_state['y_test_XGB_PCA'] = y_test           
                    st.markdown("모델이 종료되었습니다.")

############################################################################
if model_select == 'support vector machine':
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
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()
        st.subheader('데이터 학습')
        if st.checkbox('support vector machine'):
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test, y_pred_proba = import_SVM_predict_raw(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_SVM_raw = st.button('성능 비교를 위한 데이터 전송')                       
                if data_trans_SVM_raw == True:
                    st.write('SVM 성능 데이터가 전송되었습니다.')

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
                    fpr, tpr, dtc_thresholds = roc_curve(y_test, y_pred_proba)
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

                    if data_trans_SVM_raw == True:
                        st.session_state['y_pred_SVM_raw'] = y_pred
                        st.session_state['y_test_SVM_raw'] = y_test  

                    st.markdown("모델이 종료되었습니다.")
############################################################################
                        
    if cl1 == False and cl2 == True:
        with st.spinner('Updating Report...'):
            try:
                x_train = st.session_state['x_train']
                x_test = st.session_state['x_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            except:
                st.write(':red[ 데이터를 보내주세요]')
                st.stop()

        st.subheader('데이터 학습')
        if st.checkbox('support vector machine'):
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test, y_pred_proba = import_SVM_predict_pca(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_SVM_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_SVM_pca == True:
                    st.write('SVM 성능 데이터가 전송되었습니다.')
>>>>>>> f8b2d8b54920de8b222111d96d45b7a43014439a
                    
                    col1, col2, col3 = st.columns([3,1,3])
                    with col1:
                        st.code(classification_report_text)
                    with col2:
                        st.write('Confusion Matrix:', cm)
                    with col3:
                        st.write('정확도:', round(accuracy,4))
                        st.write('정밀도:', round(precision,4))
                        st.write('재현율:', round(recall,4))
                        st.write('f1-score:', round(f1,4))

                    col4, col5 = st.columns([1,1])
                    
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
                        st.write('AUC: ', AUC)
            
                    if data_trans_DNN_pca == True:
                        DNN_pca_metric = {'DNN_pca' : [accuracy, precision, recall, f1, AUC]}
                        st.session_state['DNN_pca_metric'] = DNN_pca_metric

                    model_DNN_pca = st.button('예측을 위한 모델 전송')
                    if model_DNN_pca == True:
                        st.session_state['DNN_pca'] = DNN_pca
                        st.markdown("모델이 전송되었습니다.")

############################################################################    