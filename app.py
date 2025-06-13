import streamlit as st
from process import *

work_idx, work_name, f_idx, work_f_idx, work_f_name, work_info, lecture_idx, lecture_name, WORK_K, WORK_COUNT, LECTURE_K, LECTURE_COUNT = load_data()
client = load_gpt()
model = load_emb()


with open("intro.md", "r", encoding="utf-8") as f:
    intro_md = f.read()

detail_flag = False

with st.container():

    user_input = st.chat_input('무엇을 하고 싶으신가요?')

    if user_input:
        user = st.chat_message("user")
        
        user.write(user_input)

        with st.status("처리 중...", expanded=True) as status:
            st.write("사용자 입력 임베딩 중...")
            work_query = query_process([user_input], model)
            
            st.write("직업과의 유사도 계산 중...")
            work_name_out, work_sim = work_process(work_query, WORK_K, WORK_COUNT, work_idx, work_name, f_idx, work_f_idx, work_f_name)
        
            st.write("직업을 기반으로 강의와의 유사도 계산 중...")
            lecture_query = make_lecture_query(work_name_out, work_info, model)
            lecture_name_out, lecture_sim = lecture_process(lecture_query, LECTURE_K, LECTURE_COUNT, lecture_idx, lecture_name)
            
            st.write("답변 생성 중...")
            prompt = create_prompt(user_input, work_name_out, work_sim, lecture_name_out, lecture_sim)
            gpt_out = make_output(prompt, client)

            status.update(label="✅ 처리 완료!", state="complete", expanded=False)

        ai = st.chat_message("ai")
        ai.markdown(gpt_out)

        detail_flag = True

intro, detail = st.tabs(["프로젝트 소개", "상세 결과"])

with intro:
    st.markdown(intro_md)
with detail:
    if detail_flag:
        st.write('추천된 직업')
        st.write(work_name_out)
        st.write('직업 유사도')
        st.write([round(float(s), 3) for s in work_sim])
        st.write('추천된 강의')    
        st.write(lecture_name_out)
        st.write('강의 유사도')
        st.write([round(float(s), 3) for s in lecture_sim])
