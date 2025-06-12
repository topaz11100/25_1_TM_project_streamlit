import streamlit as st
import openai
from process import load_data, query_process, work_process, make_lecture_query, lecture_process, create_prompt, make_output, WORK_K, LECTURE_K, WORK_COUNT, LECTURE_COUNT

model, work_name, work_info, lecture_name, work_idx, f_idx, work_f_idx, work_f_name, lecture_idx = load_data()

#지피티 클라이언트
API = st.secrets['API']
client = openai.OpenAI(api_key=API)

with open("intro.md", "r", encoding="utf-8") as f:
    intro_md = f.read()
detail_result = ''

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
        
        work_sim, lecture_sim = [float(round(x, 3)) for x in work_sim], [round(x, 3) for x in lecture_sim]

        detail_result = (
            f'추천된 직업 : {work_name_out}\n'
            f'직업 유사도 : {work_sim}\n'
            f'추천된 강의 : {lecture_name_out}\n'
            f'강의 유사도 : {lecture_sim}\n'
        )

        ai = st.chat_message("ai")
        ai.markdown(gpt_out)

intro, detail = st.tabs(["프로젝트 소개", "상세 결과"])

with intro:
    st.markdown(intro_md)
with detail:
    st.markdown(detail_result)

