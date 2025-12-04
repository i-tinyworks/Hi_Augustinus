📘 Hi Augustine — 어거스틴 RAG 신학 챗봇

Cerebras LLM + OpenAI Embedding + Supabase Vector DB 기반 RAG 챗봇

📖 프로젝트 소개

Hi Augustine은 히포의 어거스틴(Augustine of Hippo)의 신학 사상과 문헌을
AI 기반 RAG 시스템(Retrieval-Augmented Generation) 으로 재구성한 챗봇입니다.

사용자가 신앙·신학 질문을 하면:

OpenAI 임베딩(text-embedding-3-large)

Supabase Vector DB에서 Augustine 문헌 검색 (documents 테이블)

Cerebras LLM(gpt-oss-120b 등)으로 맥락 기반 답변 생성

을 거쳐,
마치 실제 어거스틴과 대화하는 것처럼,
따뜻하고 지혜로운 답변을 제공하는 챗봇입니다.

🎯 주요 기능
✔ 1. RAG 기반 Augustine 신학 답변

Supabase에 저장된 Augustine 문헌(Confessions, Doctrine 등)을 기반으로 답변

문헌에 없는 내용은 "본문에는 없습니다." 라고 정확히 응답

✔ 2. Cerebras 언어 모델 선택 기능

GPT-OSS 120B

QWen 32B

LLaMA 3.1 8B

사용자가 Sidebar에서 즉시 모델 변경 가능

✔ 3. OpenAI Embeddings 기반 정교한 검색

text-embedding-3-large 사용

질문 의도에 가장 가까운 Augustine 문헌 단락을 Supabase에서 검색

✔ 4. Supabase 연결 상태 표시

좌측 Sidebar에서 실시간 연결 여부 확인

🟢 연결됨

🔴 실패 (에러 메시지 표시)

✔ 5. 어거스틴 스타일 답변 생성

따뜻함 + 철학적 깊이 + 신학적 진리

비기독교인도 포용

마지막 문장에 라틴어 한 문장 요약

🏗 기술 스택
영역	기술
LLM	⭐ Cerebras gpt-oss-120b / QWen 32B / LLaMA
Embedding	OpenAI text-embedding-3-large
Vector DB	Supabase (PGVector)
Backend	Python
Frontend	Streamlit
RAG	Custom match_documents 함수 사용
📁 프로젝트 구조
project/
│── main.py                # Streamlit 챗봇 메인 코드
│── ingest.py              # PDF → chunk → embedding → Supabase 저장
│── requirements.txt
│── .env                   # API keys 저장
└── README.md

🔧 설치 & 실행
1) 저장소 클론
git clone https://github.com/사용자/hi-augustinus.git
cd hi-augustinus

2) 필요한 패키지 설치
pip install -r requirements.txt


requirements.txt 예시:

streamlit
openai
supabase
python-dotenv
pypdf

3) .env 파일 설정

프로젝트 루트에 .env 파일 생성:

CEREBRAS_API_KEY=your_cerebras_key
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_key

4) RAG 데이터 ingestion (문헌 업로드)
python ingest.py --file confessions.pdf

5) 챗봇 실행
streamlit run main.py

💡 사용 방법

좌측에서 언어 모델 선택

상태에서 Supabase 연결 확인

질문 입력:

예: “인간의 의지는 어떻게 변화되는가?”

예: “회심이란 무엇인가?”

챗봇은 Augutine 문헌에서 관련 내용을 검색하고
그 기반 위에 답변을 생성함.

🧠 어거스틴 답변의 특징

따뜻한 공감

철학·신학의 깊이

은혜, 사랑, 내적 성찰 중심

비기독교인도 환영

명료하고 이해 쉽게 설명

마지막 문장에 항상 라틴어 요약

🔍 RAG 검색 설명

질문 → Embedding 생성

Supabase match_documents 함수 호출

상위 5개 문헌 chunk 선택

LLM에게 [Context] 블록으로 전달

Strict RAG 규칙 적용

context에 없으면 “본문에는 없습니다.”

📌 예시 질문

“하나님의 은혜란 무엇인가?”

“죄책감에서 어떻게 자유로워질 수 있는가?”

“삼위일체는 어떻게 이해해야 하나?”

“Confessions 내용 안에서 ‘사랑’은 무엇인가?”