import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import plotly.express as px
from sklearn import metrics
from tool.model.DL_model import dnn
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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
def import_logistic_predict_raw(x_train, x_test ,y_train ,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

@st.cache_resource(ttl=24*60*60)
def import_logistic_predict_pca(x_train, x_test ,y_train ,y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_randomforest_predict_raw(x_train, x_test ,y_train ,y_test):
    model = RandomForestClassifier(n_estimators = 200, criterion='entropy', bootstrap=True, random_state=42, max_depth=10,min_samples_leaf=5,min_samples_split=2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

@st.cache_resource(ttl=24*60*60)
def import_randomforest_predict_pca(x_train, x_test ,y_train ,y_test):
    model = RandomForestClassifier(n_estimators = 200, criterion='entropy', bootstrap=True, random_state=42, max_depth=10,min_samples_leaf=5,min_samples_split=2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_dnn_predict_raw(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    
    model = dnn(x_train.shape[1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min')]
    progress_bar_fit = st.progress(0)
    progress_text_fit = st.empty()

    for epoch in range(100):
        model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), batch_size=64, callbacks=callbacks, verbose=0)
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
        progress_fit = (epoch + 1) / 100
        progress_bar_fit.progress(progress_fit)
        progress_text_fit.text(f"Progress: {int(progress_fit * 100)}%")

    progress_bar_fit.empty()
    return y_pred, y_test

@st.cache_resource(ttl=24*60*60)
def import_dnn_predict_pca(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    model = dnn(x_train.shape[1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min')]
    progress_bar_fit = st.progress(0)
    progress_text_fit = st.empty()

    for epoch in range(100):
        model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), batch_size=64, callbacks=callbacks, verbose=0)
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
        progress_fit = (epoch + 1) / 100
        progress_bar_fit.progress(progress_fit)
        progress_text_fit.text(f"Progress: {int(progress_fit * 100)}%")

    progress_bar_fit.empty()
    return y_pred, y_test

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_XGB_predict_raw(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    progress_bar_fit = st.progress(0)
    progress_text_fit = st.empty()

    for epoch in range(100):
        dtrain = xgb.DMatrix(data=x_train, label = y_train)
        dval = xgb.DMatrix(data=x_val, label=y_val)
        dtest = xgb.DMatrix(data=x_test, label=y_test)

        params = {'max_depth' : 10,'eta' : 0.1, 'objective' : 'binary:logistic','eval_metric' : 'error','colsample_bytree':0.8,'min_child_weight':1,
                    'reg_lambda':1,'reg_alpha':0.1}
        num_rounds = 500

        wlist = [(dtrain, 'train'), (dval,'eval')]
        
        XGB = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist, early_stopping_rounds=100)

        y_pred = XGB.predict(dtest)
        y_pred = y_pred.reshape(-1)

        for y in range(len(y_pred)):
            if y_pred[y] >= 0.5:
                y_pred[y] = 1
            else:
                y_pred[y] = 0 
        
        y_pred = y_pred.astype(int)
        y_pred = y_pred.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        progress_fit = (epoch + 1) / 100
        progress_bar_fit.progress(progress_fit)
        progress_text_fit.text(f"Progress: {int(progress_fit * 100)}%")

    progress_bar_fit.empty()
    return y_pred, y_test

@st.cache_resource(ttl=24*60*60)
def import_XGB_predict_pca(x_train, x_val, x_test ,y_train ,y_val ,y_test):
    progress_bar_fit = st.progress(0)
    progress_text_fit = st.empty()

    for epoch in range(100):
        dtrain = xgb.DMatrix(data=x_train, label = y_train)
        dval = xgb.DMatrix(data=x_val, label=y_val)
        dtest = xgb.DMatrix(data=x_test, label=y_test)

        params = {'max_depth' : 10,'eta' : 0.1, 'objective' : 'binary:logistic','eval_metric' : 'error','colsample_bytree':0.8,'min_child_weight':1,
                    'reg_lambda':1,'reg_alpha':0.1}
        num_rounds = 500

        wlist = [(dtrain, 'train'), (dval,'eval')]
        
        XGB = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist, early_stopping_rounds=100)
    
        y_pred = XGB.predict(dtest)
        y_pred = y_pred.reshape(-1)

        for y in range(len(y_pred)):
            if y_pred[y] >= 0.5:
                y_pred[y] = 1
            else:
                y_pred[y] = 0 
        
        y_pred = y_pred.astype(int)
        y_pred = y_pred.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        progress_fit = (epoch + 1) / 100
        progress_bar_fit.progress(progress_fit)
        progress_text_fit.text(f"Progress: {int(progress_fit * 100)}%")

    progress_bar_fit.empty()
    return y_pred, y_test

############################################################################
@st.cache_resource(ttl=24*60*60)
def import_SVM_predict_raw(x_train, x_test ,y_train ,y_test):
    SVM = svm.SVC(kernel='rbf', C=3, gamma=0.25, probability=True)
    SVM.fit(x_train, y_train)
    y_pred = SVM.predict(x_test)
    y_pred_proba = SVM.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

@st.cache_resource(ttl=24*60*60)
def import_SVM_predict_pca(x_train, x_test ,y_train ,y_test):
    SVM = svm.SVC(kernel='rbf', C=3, gamma=0.25, probability=True)
    SVM.fit(x_train, y_train)
    y_pred = SVM.predict(x_test)
    y_pred_proba = SVM.predict_proba(x_test)[:,1]
    y_pred = y_pred.reshape(-1)

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            y_pred[y] = 1
        else:
            y_pred[y] = 0 

    y_pred = y_pred.astype(int)
    y_pred = y_pred.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return y_pred, y_test, y_pred_proba

############################################################################
with st.sidebar:
    choose = option_menu("", ['Model','Logistic Regression', 'Random Forest', 'DNN', 'XGBOOST', 'SVM'], icons=['bar-chart', '1-square','2-square','3-square','4-square','5-square'], menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "3!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "20px","background-color": "#D8D4C7"},
    }
    )
############################################################################

st.subheader('Model')

model_select = st.selectbox("모델 선택", ['선택', 'Logistic Regression', 'Random Forest', 'Deep Neural Network', 'XGBOOST', 'support vector machine'], index=0)

if model_select == 'Logistic Regression':
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
        if st.checkbox('Logistic regression'):
            train_start = st.checkbox('RAW 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test, y_pred_proba = import_logistic_predict_raw(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_lr_raw = st.button('성능 비교를 위한 데이터 전송')                       
                if data_trans_lr_raw == True:
                    st.write('Logistic 성능 데이터가 전송되었습니다.')

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

                    if data_trans_lr_raw == True:
                        st.session_state['y_pred_Logistic_raw'] = y_pred
                        st.session_state['y_test_Logistic_raw'] = y_test  

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
        if st.checkbox('Logistic regression'):
            train_start = st.checkbox('PCA 데이터 학습 시작')
            if train_start == True:
                y_pred, y_test, y_pred_proba = import_logistic_predict_pca(x_train, x_test , y_train, y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_lr_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_lr_pca == True:
                    st.write('Logistic 성능 데이터가 전송되었습니다.')
                    
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

                    if data_trans_lr_pca == True:
                        st.session_state['y_pred_Logistic_PCA'] = y_pred
                        st.session_state['y_test_Logistic_PCA'] = y_test                
                    st.markdown("모델이 종료되었습니다.")

############################################################################    
elif model_select == 'Random Forest':
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
            y_pred, y_test, y_pred_proba = import_randomforest_predict_raw(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')
            data_trans_rf_raw = st.button('성능 비교를 위한 데이터 전송')
            if data_trans_rf_raw == True:
                st.write('randomforest 성능 데이터가 전송되었습니다.')
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

                if data_trans_rf_raw == True:
                    st.session_state['y_pred_randomforest_raw'] = y_pred
                    st.session_state['y_test_randomforest_raw'] = y_test            
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
        train_start = st.checkbox('PCA 데이터 학습 시작')
        if train_start == True:
            y_pred, y_test, y_pred_proba = import_randomforest_predict_pca(x_train, x_test , y_train, y_test)
            st.markdown('----')
            st.subheader('성능 확인')
            pred_check = st.checkbox('성능 확인')
            data_trans_rf_pca = st.button('성능 비교를 위한 데이터 전송')
            if data_trans_rf_pca == True:
                st.write('randomforest 성능 데이터가 전송되었습니다.')

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

                if data_trans_rf_pca == True:
                    st.session_state['y_pred_randomforest_PCA'] = y_pred
                    st.session_state['y_test_randomforest_PCA'] = y_test           
                st.markdown("모델이 종료되었습니다.")

############################################################################         
elif model_select == 'Deep Neural Network':
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
                y_pred, y_test = import_dnn_predict_raw(x_train, x_val, x_test ,y_train ,y_val ,y_test)
                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인')
                data_trans_dnn_raw = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_dnn_raw == True:
                    st.write('DNN 성능 데이터가 전송되었습니다.')

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

                    if data_trans_dnn_raw == True:
                        st.session_state['y_pred_dnn_raw'] = y_pred
                        st.session_state['y_test_dnn_raw'] = y_test
                        
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
                y_pred, y_test = import_dnn_predict_pca(x_train, x_val, x_test ,y_train ,y_val ,y_test)

                st.markdown('----')
                st.subheader('성능 확인')
                pred_check = st.checkbox('성능 확인하기')
                data_trans_dnn_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_dnn_pca == True:
                    st.write('DNN 성능 데이터가 전송되었습니다.')

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
            
                    if data_trans_dnn_pca == True:
                        st.session_state['y_pred_dnn_PCA'] = y_pred
                        st.session_state['y_test_dnn_PCA'] = y_test                                  
                    st.markdown("모델이 종료되었습니다.")

############################################################################   
elif model_select == 'XGBOOST':
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
                data_trans_lr_raw = st.button('성능 비교를 위한 데이터 전송')                       
                if data_trans_lr_raw == True:
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

                    if data_trans_lr_raw == True:
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
                data_trans_lr_pca = st.button('성능 비교를 위한 데이터 전송')
                if data_trans_lr_pca == True:
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

                    if data_trans_lr_pca == True:
                        st.session_state['y_pred_SVM_PCA'] = y_pred
                        st.session_state['y_test_SVM_PCA'] = y_test                
                    st.markdown("모델이 종료되었습니다.")

############################################################################    