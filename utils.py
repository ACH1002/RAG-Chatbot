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
from langchain_community.document_loaders import BSHTMLLoader
import os
import re
from bs4 import BeautifulSoup, SoupStrainer
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

# 시간 측정 데코레이터
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} took {te-ts} sec')
        return result
    return timed

# 데이터 로드 함수

@timeit
@st.cache_data
def load_data():
    ime_base_loader = WebBaseLoader(
        web_paths=[
            "https://ime.inu.ac.kr/ime/3088/subview.do", 
            "https://ime.inu.ac.kr/ime/3087/subview.do", 
            "https://ime.inu.ac.kr/ime/3086/subview.do", 
            "https://ime.inu.ac.kr/ime/3090/subview.do"
        ],
        bs_kwargs=dict(parse_only=SoupStrainer("article", attrs={"id": ["_contentBuilder"]}))
    )

    notion_loader = WebBaseLoader(web_paths=["https://ach1002.github.io/"])

    def load_all_html_files(directory):
        all_documents = []
        bs_kwargs = {'features': "lxml", 'parse_only': SoupStrainer("article", attrs={"id": ["_contentBuilder"]})}

        for filename in os.listdir(directory):
            if filename.endswith('.html'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                soup_meta = BeautifulSoup(html_content, 'lxml')
                source_meta_tag = soup_meta.find('meta', attrs={'name': 'source'})
                source_url = source_meta_tag['content'] if source_meta_tag else "Unknown"

                loader = BSHTMLLoader(file_path, bs_kwargs=bs_kwargs)
                documents = loader.load()

                for document in documents:
                    document.metadata['source'] = source_url

                all_documents.extend(documents)
        return all_documents

    pdf_loader = PyPDFDirectoryLoader("/Users/anchanho/Desktop/2024-1/system/data")
    food_loader = WebBaseLoader(
        "https://inucoop.com/main.php?mkey=2&w=4",
        bs_kwargs=dict(parse_only=SoupStrainer(class_="detail_left"))
    )

    with ThreadPoolExecutor() as executor:
        inu_data_future = executor.submit(load_all_html_files, "htmldata")
        pdf_data_future = executor.submit(pdf_loader.load)
        ime_base_data_future = executor.submit(ime_base_loader.load)
        notion_data_future = executor.submit(notion_loader.load)
        food_data_future = executor.submit(food_loader.load)

    data = (inu_data_future.result() + pdf_data_future.result() + 
            ime_base_data_future.result() + notion_data_future.result() + 
            food_data_future.result())

    return data

# 데이터 새로고침 함수
@timeit
@st.cache_data
def refresh_data_on_demand(_fs, _agent, _memory):

    new_data = load_data()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(new_data)

    hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(hf, _fs, namespace="sentence-transformer")
    faiss_vector_store = FAISS.from_documents(documents=all_splits, embedding=cached_embedder)

    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k":5})
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weight=[0.8, 0.2])
    updated_tool = create_retriever_tool(ensemble_retriever, "INU-Notice-Service", "Searches and returns documents regarding the notice service guide")
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

# 유효한 URL 검사 함수
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# 응답 포맷팅 함수
def format_response(text):
    paragraphs = re.split(r"\n{2,}", text)
    formatted_text = ""
    for paragraph in paragraphs:
        formatted_text += f"<p>{paragraph.strip()}</p>"
    return formatted_text

# 점진적 텍스트 표시 함수
def display_gradually(text, placeholder):
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(f"**Assistant:** {display_text}", unsafe_allow_html=True)
        time.sleep(0.03)