import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.python.client import device_lib
import os
import pandas as pd
#device_lib.list_local_devices()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn import metrics
from sklearn.decomposition import PCA
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
model_list = ['로지스틱 회귀_raw','로지스틱 회귀_PCA', '서포트벡터머신_raw','서포트벡터머신_PCA', '랜덤 포레스트_raw', '랜덤 포레스트_PCA', 'XGBOOST_raw' ,'XGBOOST_PCA', 'Deep Neural Network_raw', 'Deep Neural Network_PCA']
performance_select = st.multiselect('성능 확인을 위한 모델 선택', model_list)
selected_models = []
############################################################################
if '로지스틱 회귀_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            LR_raw_metric = st.session_state['LR_raw_metric']
        except:
            st.write(':red[ Logistic_raw 성능 데이터를 보내주세요]')
            st.stop()
            
    df_LR_raw = pd.DataFrame(LR_raw_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_LR_raw)

if '로지스틱 회귀_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            LR_pca_metric = st.session_state['LR_pca_metric']
        except:
            st.write(':red[ Logistic_pca 성능 데이터를 보내주세요]')
            st.stop()
            
    df_LR_pca = pd.DataFrame(LR_pca_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_LR_pca)
    
############################################################################
if '서포트벡터머신_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            SVM_raw_metric = st.session_state['SVM_raw_metric']
        except:
            st.write(':red[ SVM_raw 성능 데이터를 보내주세요]')
            st.stop()
            
    df_SVM_raw = pd.DataFrame(SVM_raw_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_SVM_raw)

if '서포트벡터머신_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            SVM_pca_metric = st.session_state['SVM_pca_metric']
        except:
            st.write(':red[ SVM_pca 성능 데이터를 보내주세요]')
            
    df_SVM_pca = pd.DataFrame(SVM_pca_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_SVM_pca)
    
############################################################################

if '랜덤 포레스트_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            RF_raw_metric = st.session_state['RF_raw_metric']
        except:
            st.write(':red[ randomforest_raw 성능 데이터를 보내주세요]')
            st.stop()
            
    df_RF_raw = pd.DataFrame(RF_raw_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_RF_raw)

if '랜덤 포레스트_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            RF_pca_metric = st.session_state['RF_pca_metric']
        except:
            st.write(':red[ randomforest_pca 성능 데이터를 보내주세요]')
            st.stop()
            
    df_RF_pca = pd.DataFrame(RF_pca_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_RF_pca)
    
############################################################################
if 'XGBOOST_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            XGB_raw_metric = st.session_state['XGB_raw_metric']
        except:
            st.write(':red[ XGB_raw 성능 데이터를 보내주세요]')
            st.stop()
            
    df_XGB_raw = pd.DataFrame(XGB_raw_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_XGB_raw)

if 'XGBOOST_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            XGB_pca_metric = st.session_state['XGB_pca_metric']
        except:
            st.write(':red[ XGB_pca 성능 데이터를 보내주세요]')
            st.stop()
            
    df_XGB_pca = pd.DataFrame(XGB_pca_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_XGB_pca)
    
############################################################################
if 'Deep Neural Network_raw' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            DNN_raw_metric = st.session_state['DNN_raw_metric']
        except:
            st.write(':red[ DNN_raw 성능 데이터를 보내주세요]')
            st.stop()
            
    df_DNN_raw = pd.DataFrame(DNN_raw_metric, index = ['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_DNN_raw)

if 'Deep Neural Network_PCA' in performance_select:
    with st.spinner('Updating Report...'):
        try:
            DNN_pca_metric = st.session_state['DNN_pca_metric']
        except:
            st.write(':red[ DNN_pca 성능 데이터를 보내주세요]')
            st.stop()
            
    df_DNN_pca = pd.DataFrame(DNN_pca_metric, index=['정확도','정밀도','재현율','F1 점수', 'AUC'])
    selected_models.append(df_DNN_pca)

############################################################################
if selected_models:
    df_concat = pd.concat(selected_models, axis=1)
    df_concat = df_concat.T
    st.table(df_concat.style.set_properties(**{'text-align': 'left','font-size': '15px','width': '160px','height': '50px'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '1.5em')]}]))

############################################################################
st.subheader('Predict')
file_path = 'dataset/pred/'
file = os.listdir(file_path)
try:
    data_list = file
    data_select = st.multiselect('예측에 사용할 하나의 데이터 선택', data_list)

    pre_df = pd.read_csv(os.path.join(file_path,data_select[0]))
    
    pre_new_df = pre_df.copy()

    if st.session_state['TO_NUM'] == 1:
        for col in pre_new_df.columns:
            if pre_new_df.dtypes[col] == "O":
                uni = pre_new_df[col].unique()
                for i, val in enumerate(uni):
                    pre_new_df[col] = pre_new_df[col].replace(val, i)

    scaler = st.session_state['scaler']
    scaler.fit(pre_new_df)
    scaled_arr = scaler.transform(pre_new_df)
    pre_new_df = pd.DataFrame(scaled_arr, columns=pre_new_df.columns)
    st.write(pre_new_df)
    
except:
    st.write('데이터를 선택해 주세요.')

pred_model_list = ['로지스틱 회귀_raw','로지스틱 회귀_PCA', '서포트벡터머신_raw','서포트벡터머신_PCA', '랜덤 포레스트_raw', '랜덤 포레스트_PCA', 'XGBOOST_raw' ,'XGBOOST_PCA', 'Deep Neural Network_raw', 'Deep Neural Network_PCA']
pred_select = st.multiselect('예측에 사용할 하나의 모델 선택', pred_model_list)

############################################################################
try:
    if '로지스틱 회귀_raw' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                LR_raw = st.session_state['LR_raw']
            except:
                st.write(':red[ 로지스틱 회귀_raw 모델을 보내주세요]')
                st.stop()

            pred_data = pre_new_df.to_numpy()

            pred = LR_raw.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if '로지스틱 회귀_PCA' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                LR_pca = st.session_state['LR_pca']
            except:
                st.write(':red[ 로지스틱 회귀_pca 모델을 보내주세요]')

            compo = st.session_state['compo']

            x_data = pre_new_df

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(pre_new_df)
            pre_new_df = pd.DataFrame(data=printcipalComponents, columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])

            pred_data = pre_new_df.to_numpy()

            pred = LR_pca.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if '서포트벡터머신_raw' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                SVM_raw = st.session_state['SVM_raw']
            except:
                st.write(':red[ 서포트벡터머신_raw 모델을 보내주세요]')
                st.stop()

            pred_data = pre_new_df.to_numpy()

            pred = SVM_raw.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if '서포트벡터머신_PCA' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                SVM_pca = st.session_state['SVM_pca']
            except:
                st.write(':red[ 서포트벡터머신_pca 모델을 보내주세요]')
                st.stop()
                
            compo = st.session_state['compo']

            x_data = pre_new_df

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(pre_new_df)
            pre_new_df = pd.DataFrame(data=printcipalComponents, columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])

            pred_data = pre_new_df.to_numpy()

            pred = SVM_pca.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if '랜덤 포레스트_raw' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                RF_raw = st.session_state['RF_raw']
            except:
                st.write(':red[ 랜덤 포레스트_raw 모델을 보내주세요]')
                st.stop()

            pred_data = pre_new_df.to_numpy()

            pred = RF_raw.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if '랜덤 포레스트_PCA' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                RF_pca = st.session_state['RF_pca']
            except:
                st.write(':red[ 랜덤 포레스트_pca 모델을 보내주세요]')
                st.stop()
                
            compo = st.session_state['compo']

            x_data = pre_new_df

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(pre_new_df)
            pre_new_df = pd.DataFrame(data=printcipalComponents, columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])

            pred_data = pre_new_df.to_numpy()

            pred = RF_pca.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if 'XGBOOST_raw' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                XGB_raw = st.session_state['XGB_raw']
            except:
                st.write(':red[ XGBOOST_raw 모델을 보내주세요]')
                st.stop()

            pred_data = pre_new_df.to_numpy()

            pred = XGB_raw.predict(x_test)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if 'XGBOOST_PCA' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                XGB_pca = st.session_state['XGB_pca']
            except:
                st.write(':red[ XGBOOST_pca 모델을 보내주세요]')
                st.stop()
                
            compo = st.session_state['compo']

            x_data = pre_new_df

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(pre_new_df)
            pre_new_df = pd.DataFrame(data=printcipalComponents, columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])

            pred_data = pre_new_df.to_numpy()
            
            pred = XGB_pca.predict(x_test)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
            
############################################################################
    if 'Deep Neural Network_raw' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                DNN_raw = st.session_state['DNN_raw']
            except:
                st.write(':red[ DNN_raw 모델을 보내주세요]')
                st.stop()

            pred_data = pre_new_df.to_numpy()

            pred_data = pred_data.reshape(pred_data.shape[0],-1,1)
            pred = DNN_raw.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
############################################################################
    if 'Deep Neural Network_PCA' in pred_select:
        with st.spinner('Updating Report...'):
            try:
                DNN_pca = st.session_state['DNN_pca']
            except:
                st.write(':red[ DNN_pca 모델을 보내주세요]')
                st.stop()
                
            compo = st.session_state['compo']

            x_data = pre_new_df

            do_pca = PCA(n_components=compo)
            printcipalComponents = do_pca.fit_transform(pre_new_df)
            pre_new_df = pd.DataFrame(data=printcipalComponents, columns = [f'PC{num+1}' for num in range(len(printcipalComponents[0]))])

            pred_data = pre_new_df.to_numpy()

            pred_data = pred_data.reshape(pred_data.shape[0],-1,1)
            pred = DNN_pca.predict(pred_data)

            pred = pred.reshape(-1)

            for y in range(len(pred)):
                    if pred[y] >= 0.5:
                        pred[y] = 1
                    else:
                        pred[y] = 0

            pre_df['error'] = pred.astype('int')

            if 1 in pred:
                st.write('이 제품은 불량품이 될 가능성이 높습니다.')
            else:
                st.write('이 제품은 양품일 가능성이 높습니다.')

            st.write(pre_df)
############################################################################
except:
    st.write('데이터와 모델을 선택해주세요')