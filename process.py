import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai
import streamlit as st

@st.cache_data
def load_data():
    # 직업 관련 파일
    # 설명 인덱스, 이름 라벨
    work_idx = faiss.read_index('data/work/work.index')
    work_name = pd.read_csv('data/work/work_name.csv')
    
    #지표기반 인덱스, 이름 라벨
    f_idx = faiss.read_index('data/feature_work/F.index')
    work_f_idx = faiss.read_index('data/feature_work/work_F.index')
    work_f_name = pd.read_csv('data/feature_work/work_f_name.csv')
    
    #이름-설명 쌍
    work_info = pd.read_csv('data/work/work_info.csv')
    
    # 강좌 관련 파일
    lecture_idx = faiss.read_index('data/lecture/lecture.index')
    lecture_name = pd.read_csv('data/lecture/lecture_name.csv')

    #뽑을 개수
    WORK_K, WORK_COUNT, LECTURE_K, LECTURE_COUNT = 20, 5, 20, 10

    return work_idx, work_name, f_idx, work_f_idx, work_f_name, work_info, lecture_idx, lecture_name, WORK_K, WORK_COUNT, LECTURE_K, LECTURE_COUNT

@st.cache_resource
def load_gpt():
    API = st.secrets['API']
    return openai.OpenAI(api_key=API)

def load_emb():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return model

#질의문 생성
def query_process(query, model):
    #쿼리 전처리
    query = model.encode(query, show_progress_bar=False, batch_size=128).astype('float32')
    faiss.normalize_L2(query)
    #반환
    return query

#직업 설명 뽑기
def make_lecture_query(name_list, work_info, model):
    #뽑기
    string = work_info.loc[work_info['직업_이름'].isin(name_list), '문장'].values[0]
    string = string.split('\n')
    #2개씩 묶기
    paired = []
    for i in range(0, len(string), 2):
        if i + 1 < len(string):
            pair = string[i] + ' ' + string[i + 1]
        else:
            pair = string[i]  # 마지막 홀수 문장은 단독으로
        paired.append(pair)
    #임베딩
    return query_process(paired, model)

#리스트에서 빈도, 유사도 기준 상위 k개 뽑기
def top_k_list(name, sim, count):
    data = dict()
    for s, n in zip(sim, name):
        if n in data:
            data[n].append(s)
        else:
            data[n] = [s]

    data = [(idx, s) for idx, s in data.items()]
    #빈도 내림차순, 경합시 유사도 내림차순
    data = sorted(data, key = lambda x : (-len(x[1]), -x[1][0]))
    #count개 추출
    data = data[:count]
    
    name_result = [n for n, s in data]
    #평균
    sim_result = [sum(s)/len(s) for n, s in data]

    return name_result, sim_result

#설명기반
#직업 처리, 강좌용 질의문 생성
def work_e_process(query, k, count, work_idx, work_name):
    #파이스 검색 후 상위만 뽑기
    sim, I = work_idx.search(query, k)
    #이름 뽑기
    name = work_name.iloc[I[0], 0].tolist()

    #빈도, 유사도 기준 상위 k개 뽑기
    name, sim = top_k_list(name, sim[0], count)

    #반환
    return name, sim

#지표기반
def work_F_process(query, count, f_idx, work_f_idx, work_f_name):
    #사용자 입력으로 지표 유사벡터 뽑기
    f_sim, I = f_idx.search(query, f_idx.ntotal)
    #관련없는거 0처리, 쿼리 형식으로 변경
    f_sim[f_sim <= 0.2] = 0
    f_sim = f_sim.astype('float32')
    
    faiss.normalize_L2(f_sim)
    
    #직업-지표점수 벡터로 검색
    sim, I = work_f_idx.search(f_sim, count)
    #이름 뽑기
    name = work_f_name.iloc[I[0], 0].tolist()
    
    #반환
    return name, sim[0]

#직업 전체과정
def work_process(query, k, count, work_idx, work_name, f_idx, work_f_idx, work_f_name):
    e_name, e_sim = work_e_process(query, k, count, work_idx, work_name)
    F_name, F_sim = work_F_process(query, count, f_idx, work_f_idx, work_f_name)

    comp = e_sim[0]
    if comp >= 0.55:
        return e_name, e_sim
    elif comp <= 0.44:
        return F_name, F_sim
    else:
        name, sim = e_name + F_name, list(e_sim) + list(F_sim)
        #결합 후 정렬
        sorted_idx = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)
        name = [name[i] for i in sorted_idx]
        sim = [sim[i] for i in sorted_idx]
        #상위 count개 뽑기
        name, sim = name[:count], sim[:count]

        return name, sim

#강좌 처리
def lecture_process(query, k, count, lecture_idx, lecture_name):
    #파이스 검색
    sim, I = lecture_idx.search(query, k)
    #모두 같은 질의이므로 펼치기
    sim, I = sim.reshape(-1), I.reshape(-1)
    name = lecture_name.iloc[I, 0].tolist()
    #상위 count개 뽑기
    name, sim = top_k_list(name, sim, count)
    
    return name, sim

# GPT 프롬프트 생성 함수
def create_prompt(user_input, work_name_out, work_sim, lecture_name_out, lecture_sim):
    job = ', '.join(
        f"{name} (유사도: {sim:.2f})" for name, sim in zip(work_name_out, work_sim)
    )
    lecture = ', '.join(
        f"{name} (유사도: {sim:.2f})" for name, sim in zip(lecture_name_out, lecture_sim)
    )

    prompt = (
        f"사용자의 입력: {user_input}\n"
        f"이 입력을 바탕으로 아래와 같은 직업들이 추천되었습니다:\n{job}\n"
        f"추천된 직업을 바탕으로 다음과 같은 강좌들이 진로에 도움이 될 수 있습니다:\n{lecture}\n"
        f"위 내용을 바탕으로 사용자의 진로 목표를 돕기 위한 요약과 조언을 작성해 주세요.\n"
        f"친절하고 구체적으로, 마치 커리어 코치가 설명하듯 도와주세요."
    )
    return prompt

#지피티 출력
def make_output(prompt, client):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "당신은 진로 추천 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content
    return output