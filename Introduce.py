import streamlit as st
from  PIL import Image

# 페이지 너비 조정 
st.set_page_config(
    page_title="기말 프로젝트", # '회사명또는팀명 - 프로젝트명'이 좋을 것 같은디
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
    </style>
    """,
    unsafe_allow_html=True,
)

sidebar_style = """
<style>
.sidebar .markdown-text-container {
    text-align: center;
}
</style>
"""
############################################################################
markdown_text = """ # 오픈채팅방
    https://open.kakao.com/me/MoZi_AI
"""
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown('')
st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)
st.sidebar.markdown(markdown_text)
############################################################################
st.title('CNC 가공 공정 데이터 기반 가공 품질 예측') # CNC 가공 공정 데이터 기반 가공 품질 예측
st.markdown('----')
# header
st.header('실습 데이터') ######
####################################
st.write('전체 실험 샘플의 결과 데이터 파일 1개 + 각 실험 샘플의 상세 데이터 파일 25개')
st.write('가공 완료된 부품 중 육안 검사를 통과한 것을 양품으로, 그렇지 아니한 제품을 불량품으로 정의한다.')
st.write('X축, Y축, Z축 위에서의 위치, 속도, 가속도 및 전류, 전압, 전원 등 상세 데이터 파일의 48개 열을 독립 변수로 설정할 수 있다.')
######################################################
with open("dataset.zip", "rb") as z:
    btn = st.download_button(
        label="실습 데이터 다운로드 (dataset.zip)", 
        data=z,
        file_name="dataset.zip",
        mime="dataset/zip"
    )
st.markdown('----')
# subheader
st.header('방법론')
# layout(checkbox 만들기)
col1, col2, col3 = st.columns([2, 2, 2])
col4, col5, col6 = st.columns([2, 2, 2])
col7, col8, col9 = st.columns([2, 2, 2])
cl1 = col1.checkbox('데이터 전처리')
cl2 = col2.checkbox('모델 ①: 로지스틱 회귀')
cl3 = col3.checkbox('모델 ②: 랜덤 포레스트')
cl4 = col4.checkbox('모델 ③: 심층 신경망')
cl5 = col5.checkbox('모델 ④: XGBoost')
cl6 = col6.checkbox('모델 ⑤: 서포트 벡터 머신')
cl7 = col7.checkbox('모델 성능 확인')

if cl1 == True: # 데이터 전처리
    st.markdown('----')
    st.subheader('① 범주형 변수 인코딩')
    st.write("'yes', 'no' 같은 범주형 데이터를 컴퓨터가 이해하기 쉬운 정수형으로 변환한다.")
    
    st.subheader('② 데이터 레이블링')
    st.write('종속 변수가 없던 데이터에 레이블을 추가한다.')

    st.subheader('③ 표준화')
    st.write("각 독립 변수를 평균은 0, 분산은 1이 되도록 변환한다.")

    st.subheader('④ 주성분 분석(principal components analysis, PCA)')
    st.write("기존의 변수를 조합하여 새로운 변수를 추출하는 기법. 고차원 데이터를 저차원 데이터로 요약한다.")

if cl2 == True: # 로지스틱
    st.markdown('----')
    st.subheader('로지스틱 회귀(Logistic Regression)')
    st.write("시그모이드 함수를 사용하여 독립 변수를 기반으로 제품의 양품 여부를 예측한다.")
    show_content = False
    with st.expander('동영상 시청'):
        st.video("LogisticRegression.mp4", format='video/mp4')
        st.markdown("[출처]https://www.youtube.com/watch?v=9q0gY_QxBeA")

if cl3 == True: # 랜덤 포레스트
    st.markdown('----')
    st.subheader('랜덤 포레스트(Random Forest)')
    st.write('의사 결정 나무의 앙상블을 사용하여 여러 독립 변수와 상호 작용을 고려하여 제품을 양품 또는 불량품으로 분류한다.')
    show_content = False
    with st.expander('동영상 시청'):
        show_content = not show_content
        st.video("RandomForest.mp4", format='video/mp4')
        st.markdown("[출처]https://www.youtube.com/watch?v=gkXX4h3qYm4")

if cl4 == True: # DNN
    st.markdown('----')
    st.subheader('심층 신경망(Deep Neural Network, DNN)')
    st.write('연결된 여러 계층의 노드를 사용하여 패턴을 학습하고 독립 변수를 기반으로 제품을 양품 또는 불량품으로 분류한다.')
    show_content = False
    with st.expander('동영상 시청'):
        show_content = not show_content
        st.video("DNN(신경망의 학습과정, 경사하강법).mp4", format='video/mp4')
        st.markdown("[출처]https://www.youtube.com/watch?v=Ei_md7n40mA")

if cl5 == True: # XGBoost
    st.markdown('----')
    st.subheader('XGBoost(eXtreme Gradient Boosting)') 
    st.write('의사 결정 나무의 앙상블을 반복적으로 구성하여 독립 변수를 기반으로 제품의 양품 여부를 예측한다.')
    show_content = False
    with st.expander('동영상 시청'):
        show_content = not show_content
        st.video("XGBoost.mp4", format='video/mp4')
        st.markdown("[출처]https://www.youtube.com/watch?v=yw-E__nDkKU")

if cl6 == True: # SVM
    st.markdown('----')
    st.subheader('서포트 벡터 머신(support vector machine, SVM)') 
    st.write('고차원 공간에서 최적의 초평면을 찾아 마진을 최대화하여 독립 변수를 분류하고 양품과 불량을 예측한다.')
    show_content = False
    with st.expander('동영상 시청'):
        show_content = not show_content
        st.video('SVM.mp4', format='video/mp4')
        st.markdown("[출처]https://www.youtube.com/watch?v=_YPScrckx28")

if cl7 == True: # Confusion Matrix
    st.markdown('----')
    st.subheader('혼동 행렬(Confusion Matrix)')
    st.write('분류 알고리즘을 평가하기 위해 분류 결과를 시각화한 표')
    from PIL import Image

    col8, col9 = st.columns([2,2])
    with col8:  
        image = Image.open("ConfusionMatrix.webp")
        new_width = 500
        new_height = 400

        resized_image = image.resize((new_width, new_height))

        st.image(resized_image)
    with col9:
        st.write(r'① 정확도(Accuracy) = $\frac{(TP + TN)}{(TP + TN + FP + FN)}$')
        st.write('전체 사건 중 정확하게 예측한 것의 비율을 의미한다.')
        st.write('\n')
        st.write(r'② 정밀도(Precision) = $\frac{TP}{(TP + FP)}$')
        st.write('양성으로 예측된 결과 중 정확한 예측의 비율을 의미한다.')
        st.write('<p style="font-size:20px;"></p>', unsafe_allow_html=True)
        st.write(r'③ 재현율(Recall) = $\frac{TP}{(TP + FN)}$')
        st.write('실제 양성 중 모델이 정확하게 양성으로 예측한 비율을 나타낸다.')
        st.write('<p style="font-size:20px;"></p>', unsafe_allow_html=True)
        st.write(r'④ F1 Score = 2*$\frac{(Precision * Recall)}{(Precision + Recall)}$')
        st.write('정밀도와 재현율의 조화평균')

    st.markdown('') # 여백의 미
        
    st.subheader('ROC 곡선(receiver operating characteristic curve, ROC curve)') # ROC
    st.write('모든 분류 임계값에서 모델의 성능을 보여주는 그래프')
    from PIL import Image

    col10, col11 = st.columns([2,2])
    with col10:  
        image = Image.open("ROCcurve.png")
        new_width = 500
        new_height = 400

        resized_image = image.resize((new_width, new_height))

        st.image(resized_image)
    with col11:
        st.write("아 이거 겁나 헷갈리네특이도는 실제 음성 중 모델이 정확하게 음성으로 예측한 것의 비율을 나타낸다.")
        st.write("민감도는")
        st.write("X축은 '1 - 특이도'를 의미하고 Y축은 '민감도'를 의미한다.")
        

# 회사 소개

