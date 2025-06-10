import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')
st.title('🤖 전세사기피해 상담 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []


## 이전 채킹 내용 화면 출력
for msg in st.session_state.message_list:
    with st.chat_message(msg['role']):
        st.write(msg['content'])


## 채팅 메시지 =======================================================================
placeholder = '전세사기피해와 관련된 궁금한 내용을 질문하세요.'
if user_question := st.chat_input(placeholder=placeholder): ## prompt 창
    ## 사용자 메시지 ######################################
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})


    # AI 메시지 ###########################################
    # spinner 추가
    with st.spinner('잠시만 기다려주세요:)'):
        ai_message = get_ai_message(user_question)

    with st.chat_message('ai'):
        ## AI 메시지 화면 출력
        st.write(ai_message)
    st.session_state.message_list.append({'role': 'ai', 'content': ai_message})

print(f'after: {st.session_state.message_list}') 






