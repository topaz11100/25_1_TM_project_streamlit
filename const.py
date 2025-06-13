from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)

#임베딩 모델
MODEL = 'paraphrase-multilingual-mpnet-base-v2'

#직업관련 파일
#설명기반 인덱스
WORK_IDX = BASE_DIR + '/data/work/work.index'
#이름 라벨
WORK_NAME = BASE_DIR + '/data/work/work_name.csv'

#지표기반 인덱스
#지표임베딩 인덱스
F_IDX = BASE_DIR + '/data/feature_work/F.index'
#직업-지표점수 인덱스
WORK_F_IDX = BASE_DIR + '/data/feature_work/work_F.index'
#이름 라벨
WORK_F_NAME = BASE_DIR + '/data/feature_work/work_f_name.csv'

#이름-설명 쌍
WORK_INFO = BASE_DIR + '/data/work/work_info.csv'

#강좌관련 파일
#인덱스
LECTURE_IDX = BASE_DIR + '/data/lecture/lecture.index'
#이름 라벨
LECTURE_NAME = BASE_DIR + '/data/lecture/lecture_name.csv'

#문서 DB에서 몇 개 뽑을지에 대한 상수
WORK_K = 20
LECTURE_K = 20
#사용자에게 표시할 직업, 강좌 개수
WORK_COUNT = 5
LECTURE_COUNT = 10

#상수 로드
def const_return():
    return MODEL, WORK_IDX, WORK_NAME, F_IDX, WORK_F_IDX, WORK_F_NAME, WORK_INFO, LECTURE_IDX, LECTURE_NAME, WORK_K, LECTURE_K, WORK_COUNT, LECTURE_COUNT