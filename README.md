# RAG-Chatbot
산업경영공학과 2024-1 졸업작품


# 주제: RAG를 활용한 LLM 기반 인천대학교 산업경영공학과 챗봇

## RAG(Retrieval Augmented Generation)

### RAG를 선택한 이유 :

1. 자연어 기반 검색 시스템이라는 주제를 고려하면 파인튜닝을 통한 범용 챗봇보다는 RAG를 통한 해당 분야에 특출난 지식을 가진 챗봇이 더 어울린다고 판단
2. 파인튜닝은 gpt나 gemini 등의 LLM에 사용 시 상당한 비용이 필요하지만 RAG의 경우 gpt의 답변 기능만 사용하기 때문에 적은 비용이 발생
3. 실제 기업에서도 QnA용 챗봇을 만들 때 RAG를 사용한 경우가 많음
    1. Skelter Labs의 사내 업무 도우미 챗봇
        
        [Skelter Labs Blog - BELLA QNA로 신제품 런칭 이벤트 챗봇 제공하기](https://www.skelterlabs.com/blog/bella-qna-launching-chatbot)
        
    2. KBPay 앱 QnA챗봇
    
    [Skelter Labs Blog - 금융업 LLM 활용 사례 :: KB국민카드](https://www.skelterlabs.com/blog/llm-usecase)
    
    복잡한 규정이 많은 금융업에서도 RAG를 활용한 챗봇을 사용하는 것을 보고 결정하게 됨
    


![Untitled](/docs/RAG.png)

미리 데이터를 벡터DB에 저장을 해둔 뒤 유저가 질문하면 임베딩 모델이 관련된 문서의 관련된 문단이나 문장을 발췌하여 LLM모델에게 질문과 같이 입력하여 답변을 얻어낸다.

## RAG 프로세스

### 전처리 단계

1. 문서 로드(URL, PDF, TXT …. )
2. 문서 청크로 나누기
3. Embedding
4. Vector DB

### 서비스 단계

1. 유저 질문
2. 전처리 단계에서 만든 Retriever 검색을 통해 원하는 부분 발췌
3. PROMPT를 통해 원하는 답변을 얻기 위한 처리
4. LLM 답변 생성

# Detail

## 1. 문서로드

- 웹페이지 = WebBaseLoader
- pdf = PyPDF

## 2. 문서 청크로 나누기

- RecursiveCharacterTextSplitter

## 3. Embedding

- HuggingFace OpenSouce(bespin-global/klue-sroberta-base-continue-learning-by-mnr)

## 4. Vector DB

- FAISS

## 5. 유저 질문

- 웹 사이트(Streamlit)

## 6. Retriever

- EnsemberRetriever(Faiss(의미적 유사성 기반) + bm25(키워드 기반))

## 7. Prompting

- langchain hub(RAG)

## 8. LLM 답변 생성

- OpenAI(gpt-3.5-turbo)
