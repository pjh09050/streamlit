import streamlit as st
import numpy as np
import pandas as pd

# 페이지 너비 조정 
st.set_page_config(
    page_title="기말 프로젝트",
    page_icon="heart",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        max-width: 1300px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# title을 출력합니다.
st.title('주제')
# header
st.header('소주제')
# subheader
st.subheader('팀원')
############################################################################
# layout(checkbox 만들기)
col1, col2, col3, col4 = st.columns([2,2,2,2])
col1.checkbox('발표팀장 : 유창연')
col2.checkbox('총괄팀장 : 서민혁')
col3.checkbox('분석팀장 : 서영석')
col4.checkbox('개발팀장 : 박진환')

############################################################################
# 사이드바를 추가합니다.
st.sidebar.header('데이터 불러오기')
# 파일 업로드
df = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if df is not None:
    try:
        # Pandas DataFrame으로 데이터 로드
        data = pd.read_csv(df)
        # 로드된 데이터 출력
        st.write('로드된 데이터와 크기:', data.shape)
        st.write(data)
    except Exception as e:
        st.write('파일을 로드할 수 없습니다.', e)

st.sidebar.header('방법론 선택')
solve1 = st.sidebar.checkbox('방법론1', key=1)
solve2 = st.sidebar.checkbox('방법론2', key=2)
solve3 = st.sidebar.checkbox('방법론3', key=3)
if solve1 is True:
    st.write('선택한 방법론 : ', '방법론1')
elif solve2 is True:
    st.write('선택한 방법론 : ', '방법론2')
else:
    st.write('선택한 방법론 : ', '방법론3')

############################################################################
# 변수 선택
# return : list
st.sidebar.header('변수 선택')
select_multi_species = st.sidebar.multiselect('확인하고 싶은 변수 선택(복수선택가능)',['forecast', 'temperature', 'windspeed', 'winddirection', 'humidity'])
# 원래 dataframe으로 부터 꽃의 종류가 선택한 종류들만 필터링 되어서 나오게 일시적인 dataframe을 생성합니다
#tmp_df = df[df['species'].isin(select_multi_species)]

# 선택한 종들의 결과표를 나타냅니다.  
#st.table(tmp_df)
############################################################################
st.sidebar.header('파라미터 튜닝')
# 첫 번째 파라미터를 입력받는 텍스트 상자
param1 = st.sidebar.text_input('첫 번째 파라미터', '')
# 두 번째 파라미터를 입력받는 텍스트 상자
param2 = st.sidebar.text_input('두 번째 파라미터', '')
# 세 번째 파라미터를 입력받는 슬라이더
param3 = st.sidebar.slider('세 번째 파라미터', 0, 100, 50)

# 입력 받은 파라미터 값을 출력
st.write('첫 번째 파라미터:', param1)
st.write('두 번째 파라미터:', param2)
st.write('세 번째 파라미터:', param3)

############################################################################
radio_select =st.sidebar.radio(
    "뭐 넣지? ",
    ['forecast', 'temperature', 'windspeed', 'winddirection', 'humidity'],
    horizontal=True
    )
# 선택한 컬럼의 값의 범위를 지정할 수 있는 slider를 만듭니다. 
slider_range = st.sidebar.slider(
    "choose range of key column",
     0.0, #시작 값 
     10.0, #끝 값  
    (2.5, 7.5) # 기본값, 앞 뒤로 2개 설정 /  하나만 하는 경우 value=2.5 이런 식으로 설정가능
)

# 필터 적용버튼 생성 
start_button = st.sidebar.button(
    "파라미터 튜닝 적용 "#"버튼에 표시될 내용"
)

# button이 눌리는 경우 start_button의 값이 true로 바뀌게 된다.
# 이를 이용해서 if문으로 버튼이 눌렸을 때를 구현 
if start_button:
    tmp_df = select_multi_species
    #tmp_df = st.write(df[df['forecast'].isin(select_multi_species)])
    #slider input으로 받은 값에 해당하는 값을 기준으로 데이터를 필터링합니다.
    #tmp_df= tmp_df[ (tmp_df[radio_select] >= slider_range[0]) & (tmp_df[radio_select] <= slider_range[1])]
    st.table(tmp_df)
    # 분석 시작 
    st.sidebar.success("Start")
    st.success("작업이 완료되었습니다!")

############################################################################
# 입력 받는 값을 출력합니다.
name = st.text_input('이름을 입력하세요')
age = st.slider('나이를 선택하세요', 0, 100)

# 입력 받은 값을 출력합니다.
st.write('이름:', name)
st.write('나이:', age)

# 차트를 출력합니다.
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])
st.line_chart(chart_data)
