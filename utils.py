from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import dotenv
dotenv.load_dotenv()
import streamlit as st
import time
import bs4
from langchain_community.document_loaders import BSHTMLLoader
import os
import re
from bs4 import BeautifulSoup, SoupStrainer

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} took {te-ts} sec')
        return result
    return timed

@timeit
@st.cache_data
def load_data():
    ime_base_loader = WebBaseLoader(
        web_paths= ["https://ime.inu.ac.kr/ime/3088/subview.do", "https://ime.inu.ac.kr/ime/3087/subview.do", "https://ime.inu.ac.kr/ime/3086/subview.do", "https://ime.inu.ac.kr/ime/3090/subview.do"],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "article",
                attrs={"id": ["_contentBuilder"]},
            )
        ),
    )
    notion_loader = WebBaseLoader(
        web_paths= ["https://ach1002.github.io/"],
        # bs_kwargs=dict(
        #     parse_only=bs4.SoupStrainer(
        #         "class",
        #         attrs={"id": ["layout-content"]},
        #     )
        # ),
    )

    # ime_specific_loader = WebBaseLoader(
    #     web_paths= ["https://url.kr/8axpe4"],
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             "div",
    #             attrs={"class": ["board-view-info", "view-con"]},
    #         )
    #     ),
    # )
    # ime_specific_loader.requests_kwargs = {"verify": False}
    def load_all_html_files(directory):
        all_documents = []
        bs_kwargs = {
            'features': "lxml",
            'parse_only': SoupStrainer(
                "article",
                attrs={"id": ["_contentBuilder"]}
            )
        }
        # 지정된 디렉토리 내 모든 파일을 순회합니다.
        for filename in os.listdir(directory):
            if filename.endswith('.html'):  # HTML 파일만 처리
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                # BeautifulSoup를 사용하여 메타 태그에서 source 정보 추출
                soup_meta = BeautifulSoup(html_content, 'lxml')
                source_meta_tag = soup_meta.find('meta', attrs={'name': 'source'})
                source_url = source_meta_tag['content'] if source_meta_tag else "Unknown"

                # 각 HTML 파일에 대해 BSHTMLLoader 인스턴스 생성 및 로드
                loader = BSHTMLLoader(file_path, bs_kwargs=bs_kwargs)
                documents = loader.load()

                # 로드된 각 문서에 메타데이터 추가
                for document in documents:
                    document.metadata['source'] = source_url

                all_documents.extend(documents)

        return all_documents


    pdf_loader = PyPDFDirectoryLoader("/Users/anchanho/Desktop/2024-1/system/data")

    # 데이터 로드
    inu_data = load_all_html_files("htmldata")
    pdf_data = pdf_loader.load()
    ime_base_data = ime_base_loader.load()
    # ime_specific_data = ime_specific_loader.load()
    notion_data = notion_loader.load()

    data = pdf_data + ime_base_data + notion_data + inu_data
    return data

from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor

@timeit
@st.cache_data
def refresh_data_on_demand(_fs, _agent, _memory):
    food_loader = WebBaseLoader(
        "https://inucoop.com/main.php?mkey=2&w=4",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("detail_left")
            )
        )
    )  # 가정: 데이터가 업데이트되는 새로운 URL
    food_data = food_loader.load()
    new_data = load_data() + food_data

    #불러온 데이터에서 글자를 청크단위로 문장과 문단을 나눔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    all_splits = text_splitter.split_documents(new_data)
    # print(f"data 로드 결과 {all_splits}")
    hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, _fs, namespace="sentence-transformer"
    )
    #컴퓨터가 알아듣기 쉽게 글자가 아닌 2진법으로 변환시킨 벡터 데이터를 저장
    faiss_vector_store = FAISS.from_documents(documents = all_splits, embedding = cached_embedder)

    #LLM이 질문을 받을 때마다 불러오는 데이터 = dense retriever + sparse retriever = ensemble retriever
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k":2})
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, faiss_retriever], weight = [0.8, 0.2]
    )   
    # 에이전트 도구 업데이트
    updated_tool = create_retriever_tool(
        ensemble_retriever,
        "INU-Notice-Service",
        "Searches and returns documents regarding the notice service guide",
    )
    # 에이전트와 실행기의 도구 리스트 직접 갱신
    updated_tools = [updated_tool]

    st.session_state.agent_executor = AgentExecutor(
        agent=_agent,
        tools=updated_tools,
        memory=_memory,
        verbose=False,
        return_intermediate_steps=True,
    )
    st.success("Data refreshed successfully!")

    st.session_state.ensemble_retriever = ensemble_retriever


from urllib.parse import urlparse
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def format_response(text):
    # 문단을 나누는 간단한 규칙: 두 개 이상의 개행을 찾아내서 HTML 문단 태그로 변경
    paragraphs = re.split(r"\n{2,}", text)
    formatted_text = ""
    for paragraph in paragraphs:
        # HTML <p> 태그를 사용하여 문단을 구분합니다.
        formatted_text += f"<p>{paragraph.strip()}</p>"
    return formatted_text

def display_gradually(text, placeholder):
    # 텍스트를 한 글자씩 점진적으로 표시합니다.
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(f"**Assistant:** {display_text}", unsafe_allow_html=True)
        time.sleep(0.03)  # 각 글자 사이의 지연 시간을 조절할 수 있습니다.

# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# def get_conversation_chain(retriever,llm):
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm, 
#             chain_type="stuff", 
#             retriever= retriever, 
#             memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
#             get_chat_history=lambda h: h,
#             return_source_documents=True,
#             verbose = True
#         )

#     return conversation_chain


