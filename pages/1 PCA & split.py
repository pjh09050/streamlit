import streamlit as st
from tensorflow.python.client import device_lib
from streamlit_option_menu import option_menu
import os
device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tool import handmade_pca
from imblearn.over_sampling import SMOTE
from tool.visualize_tool import twod_visualization, threed_visualization
import matplotlib.pyplot as plt
############################################################################
with st.sidebar:
    choose = option_menu("순서", ["데이터 확인", "변수 선택", "데이터 불균형 확인", "PCA 진행", "PCA 시각화", "데이터 셋 분할"], icons=['1-square','2-square','3-square','4-square','5-square','6-square'] ,menu_icon="bi bi-card-list",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#D8D4C7"},
    }
    )
############################################################################
with st.spinner('Updating Report...'):
    try:
        total_data = st.session_state['train_data']
    except:
        st.write('데이터를 보내주세요')
        st.stop()

data_check = st.checkbox('데이터 확인')
if data_check == True:
    st.write(total_data)

st.markdown('----')

st.subheader('PCA')
if st.checkbox('변수 선택'):
    try:
        st.markdown("\n'하나 이상의 독립변수와 종속변수 하나를 지정해주세요.")
        st.markdown("위의 보기를 확인하고 독립변수들의 번호를 x, 종속변수의 번호를 y에 선택해주세요.")
        st.markdown('\n***독립변수 선택이 어렵다면 날짜, 번호 관련 변수를 제외하고 모두 선택해주세요.***')
        all_select = st.checkbox('모든 변수 선택(선택한 y변수 제외한 모든 변수 선택)')
        want_select = st.checkbox('원하는 변수 선택')
        if all_select == True and want_select == False:
            feature_y = st.multiselect('y 변수 선택 : ', total_data.columns)
            feature_x = [col for col in total_data.columns if col not in feature_y]
            
        if all_select == False and want_select == True:
            feature_x = st.multiselect('x 변수 선택:', total_data.columns, default=total_data.columns.tolist())
            feature_y = st.multiselect('y 변수 선택 : ', total_data.columns)
        total_data_pca = handmade_pca(total_data, feature_x, feature_y)
        feature_x_columns = list(feature_x)
        feature_y_columns = list(feature_y)

        if all(column in total_data.columns for column in feature_x_columns):
            df = total_data[feature_x_columns]
            feature_y_columns_unique = list(set(feature_y_columns))

            if all(column in total_data.columns for column in feature_y_columns_unique):
                df = pd.concat([df, total_data[feature_y_columns_unique]], axis=1)
                st.write(df)

        st.markdown('----')

        if st.checkbox('데이터 불균형 확인'):
            x_data = df.iloc[:,:-1]
            y_data = df.iloc[:, -1]
            y_count = y_data.value_counts()

            for val in range(len(y_count)):
                if round(y_count[val]/len(y_data),2) != round(1/len(y_count),2):
                    T = True
                    st.markdown("데이터 불균형이 존재합니다. SMOTE를 통해 데이터 불균형을 해소하세요")
                    st.markdown('SMOTE는 최근접 이웃(k-NN) 알고리즘을 기반으로 하여 데이터를 새로 생성해 종속변수 데이터가 같은 비율을 갖게 하는 알고리즘 입니다.')
                    st.markdown('----')
                    if T == True:
                        if st.checkbox('SMOTE 진행'):
                            try:
                                x_arr = df.iloc[:,:-1].values
                                y_arr = df.iloc[:,-1].values
                                smote = SMOTE(random_state=42)
                                new_x_arr,new_y_arr = smote.fit_resample(x_arr,y_arr)

                                st.write('기존 데이터 형태: ', x_arr.shape, y_arr.shape)
                                st.write('SMOTE 적용 후 데이터 형태: ', new_x_arr.shape, new_y_arr.shape)
                                st.write('SMOTE 적용 후 종속변수 데이터 분포: ', pd.Series(new_y_arr).value_counts())
                                
                                x_df = pd.DataFrame(new_x_arr, columns=df.columns[:-1])
                                y_df = pd.DataFrame(new_y_arr, columns=[df.columns[-1]])
                                df = pd.concat([x_df,y_df], axis=1)
                    
                                st.markdown("데이터 불균형이 해소 되었습니다. 주성분분석을 진행하세요.")
                            except:
                                st.markdown('앞서 입력하신 번호가 중복되거나 범위 안에 있는지 다시 한 번 확인해주세요.')
                    break
                else:
                    st.markdown('데이터 불균형이 존재하지 않습니다.')
                    st.markdown('.pca()를 통해 주성분분석을 진행하세요.')

        st.markdown('----')
        if st.checkbox('PCA 진행'):
            try:
                col = x_data.columns
                x = StandardScaler().fit_transform(x_data)
                features = x.T
                cov_mat = np.cov(features)

                values, vectors = np.linalg.eig(cov_mat)

                explained_variances = []
                for i in range(len(values)):
                    explained_variances.append(values[i] / np.sum(values))
                explained_variances = sorted(explained_variances, reverse=True)
    
                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(explained_variances)
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Explained Variance')
                ax.set_title('Scree Plot')
                ax.set_xticks(np.arange(0, len(explained_variances), 2))
                st.pyplot(fig)

                df_for_explain = pd.DataFrame({'고유값': np.array(values), '기여율(설명력)': np.array(explained_variances)})
                df_for_explain['누적 기여율'] = df_for_explain['기여율(설명력)'].cumsum()

                idx = df_for_explain.index
                df_for_explain = df_for_explain.T
                df_for_explain.columns = [f'PC{i + 1}' for i in idx]

                for col in range(len(df_for_explain.columns)):
                    if df_for_explain.iloc[2, col] > 0.8:
                        num_PC = col + 1
                        break

                st.write(df_for_explain)
                st.markdown("\n위의 'Scree Plot'과 표를 참고하여 주성분 개수를 입력해주세요.")
                st.markdown("일반적으로 Scree Plot의 기울기가 완만해지는 시점이나 누적 기여율이 80%~90%인 시점에서 주성분 개수를 결정합니다.")
                st.markdown(f'추천 주성분 개수: {num_PC}')
                compo = st.number_input("주성분 개수:", min_value=1, value=num_PC)

                do_pca = PCA(n_components=compo)
                principalComponents = do_pca.fit_transform(x)
                pca_df = pd.DataFrame(data=principalComponents, columns=[f'PC{num + 1}' for num in range(len(principalComponents[0]))])
                
                pca_df = pd.concat([pca_df, y_data], axis=1)
                pca_df = np.round(pca_df,2)
                st.write(pca_df)
                st.write('')
                if compo == 2:
                    st.write("시각화가 가능합니다. '.pca_visualize(dimension='2d')'을 통해 확인 가능합니다.")
                if compo == 3:
                    st.write("시각화가 가능합니다. '.pca_visualize(dimension='3d')'을 통해 확인 가능합니다.")

                st.write("주성분 설정이 끝났습니다. 학습용 데이터를 생성해주세요.")

            except Exception as e:
                st.write('입력하신 주성분 개수를 다시 한 번 확인해주세요')

        st.markdown('----')
        if st.checkbox('PCA 시각화'):
            dimension = st.selectbox("차원 선택", ['2d', '3d'])
            if dimension=='2d':
                try:
                    twod_visualization(pca_df)
                    print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
                except:
                    print('데이터의 차원이 달라 2차원 시각화가 불가능 합니다.')
            elif dimension=='3d':
                try:
                    threed_visualization(pca_df)
                    print("이제 'train_split()'을 통해 학습용 데이터를 생성해주세요.")
                except:
                    print('데이터의 차원이 달라 3차원 시각화가 불가능 합니다.')
            else:
                raise Exception('파라미터를 다시 확인해주세요')
        
        st.markdown('----')
        st.subheader('데이터 셋 분할')
        if st.checkbox('데이터 셋 분할'):
            ratio1 = st.number_input('학습 데이터 비율', min_value=0.0, max_value=1.0, value=0.8, step=0.1)
            ratio2 = st.number_input('테스트 데이터 비율', min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            if ratio1 + ratio2 != 1:
                st.write(':red[비율 합이 1이 되도록 설정해주세요.]')
            else:
                x_data = pca_df.iloc[:,:len(pca_df.columns)-1].to_numpy()
                y_data = pca_df.iloc[:,-1].to_numpy()
                
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio2, random_state=1)
                #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio2, random_state=1)
                #x_train, x_val, x_test, y_train, y_val, y_test = custom_train_test_split(x_data, y_data, ratio1, ratio2, ratio3)
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
            st.session_state['x_train'] = x_train
            st.session_state['x_test'] = x_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.write('데이터가 전송되었습니다.')
    except:
        st.markdown(':red[ 변수 선택해주세요]')