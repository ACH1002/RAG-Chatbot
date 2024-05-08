from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import dotenv
dotenv.load_dotenv()
import streamlit as st
import time
import bs4
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
import os
import re
from bs4 import BeautifulSoup, SoupStrainer




# 파일에서 URL 목록 읽기
urls = []
with open('inu_urls.txt', 'r') as file:
    current_url = None
    for line in file:
        stripped_line = line.strip()
        if stripped_line:
            current_url = stripped_line
            urls.append(current_url)
        else:
            current_url = None



#url에서 데이터를 불러옴
# inu_loader = DirectoryLoader(path=os.getcwd(), glob="/htmldata/*.html", loader_cls = BSHTMLLoader)

ime_loader = WebBaseLoader(
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
ime_data = ime_loader.load()
notion_data = notion_loader.load()
print(notion_loader)

print(notion_data)

data = inu_data + pdf_data + ime_data + notion_data

# print(f"data 로드 결과 {data}")

fs = LocalFileStore("./cache/")

#불러온 데이터에서 글자를 청크단위로 문장과 문단을 나눔
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
all_splits = text_splitter.split_documents(data)
# print(f"data 로드 결과 {all_splits}")
hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    hf, fs, namespace="sentence-transformer"
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
from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)
#retriever 툴을 생성
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=ensemble_retriever, return_source_documents=True, verbose=True)

tool = create_retriever_tool(
    ensemble_retriever,
    "INU-Notice-Service",
    "Searches and returns documents regarding the notice service guide",
)
tools = [tool]

#이전 대화 기록들을 기반으로 질문에 답하기 위해 대화 기록 저장
memory_key = "history"

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)




memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
print(rag_prompt)
#프롬프팅(후처리: Hallucination 방지, 번역)
system_message = SystemMessage(
    content=(
        "You are an assistant for question-answering tasks. Answer only the questions based on the text provided. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"
    )
)
#메모리키와 시스템메세지 적용한 프롬프트
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

#프롬프트를 적용한 에이전트
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=ensemble_retriever, 
#     return_source_documents=True)

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

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = agent_executor

from urllib.parse import urlparse
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def refresh_data_on_demand():
    # 새로운 데이터 소스 로드
    custom_loader = WebBaseLoader(
        "https://ach1002.github.io/"
    )  # 가정: 데이터가 업데이트되는 새로운 URL
    custom_data = custom_loader.load()
    food_loader = WebBaseLoader(
        "https://inucoop.com/main.php?mkey=2&w=4",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("detail_left")
            )
        )
    )  # 가정: 데이터가 업데이트되는 새로운 URL
    food_data = food_loader.load()

    ime_loader = WebBaseLoader(
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
    ime_data = ime_loader.load()
    notion_data = notion_loader.load()

    new_data = inu_data + pdf_data + ime_data + notion_data + food_data

    #불러온 데이터에서 글자를 청크단위로 문장과 문단을 나눔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    all_splits = text_splitter.split_documents(new_data)
    # print(f"data 로드 결과 {all_splits}")
    hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, fs, namespace="sentence-transformer"
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
        agent=agent,
        tools=updated_tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )
    st.success("Data refreshed successfully!")

st.title("AI 인천대학교 산업경영공학과 상담원")

if st.button("refresh data") :
    refresh_data_on_demand()

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        result = st.session_state.agent_executor({"input": prompt})
        output = result['output']
        formatted_output = format_response(output)  # 출력 포매팅 함수 호출

        # 여기에 출처를 추가합니다. 출처 정보는 필요에 따라 조정하십시오.
        source_info = ensemble_retriever.get_relevant_documents(prompt)
        print(source_info[0])
        if source_info and is_valid_url(source_info[0].metadata['source']):
            formatted_output += f"<p>Source: <a href='{source_info[0].metadata['source']}' target='_blank'>출처</a></p>"
        if source_info and is_valid_url(source_info[1].metadata['source']):
            formatted_output += f"<p>Source: <a href='{source_info[1].metadata['source']}' target='_blank'>출처</a></p>"

        with st.container():
            message_placeholder = st.empty()  # 결과를 표시할 스트림릿 컨테이너 준비
            display_gradually(formatted_output, message_placeholder)  # 점진적 표시 함수 호출


    st.session_state.messages.append({"role": "assistant", "content": formatted_output})



