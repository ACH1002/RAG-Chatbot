from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
import dotenv
dotenv.load_dotenv()
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_functions_agent
from utils import load_data, refresh_data_on_demand, is_valid_url, format_response, display_gradually
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain import hub

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    data = load_data()

    st.session_state._fs = LocalFileStore("./cache/")

    #불러온 데이터에서 글자를 청크단위로 문장과 문단을 나눔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    all_splits = text_splitter.split_documents(data)
    hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        hf, st.session_state._fs, namespace="sentence-transformer"
    )
    #컴퓨터가 알아듣기 쉽게 글자가 아닌 2진법으로 변환시킨 벡터 데이터를 저장
    faiss_vector_store = FAISS.from_documents(documents = all_splits, embedding = cached_embedder)

    #LLM이 질문을 받을 때마다 불러오는 데이터 = dense retriever + sparse retriever = ensemble retriever
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k":2})
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 2
    st.session_state.ensemble_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, faiss_retriever], weight = [0.8, 0.2]
    )
    llm = ChatOpenAI(temperature=0, max_tokens=1024)
    #retriever 툴을 생성

    tool = create_retriever_tool(
        st.session_state.ensemble_retriever,
        "INU-Notice-Service",
        "Searches and returns documents regarding the notice service guide",
    )
    tools = [tool]

    #이전 대화 기록들을 기반으로 질문에 답하기 위해 대화 기록 저장
    memory_key = "history"

    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

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
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        return_intermediate_steps=True,
    )
    st.session_state.agent_executor = agent_executor


st.title("AI 인천대학교 산업경영공학과 상담원")

if st.button("refresh data") :
    refresh_data_on_demand(_fs=st.session_state._fs, _agent=st.session_state.agent_executor.agent, _memory=st.session_state.agent_executor.memory)

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
        print(result)
        output = result['output']
        formatted_output = format_response(output)  # 출력 포매팅 함수 호출

        # 여기에 출처를 추가합니다. 출처 정보는 필요에 따라 조정하십시오.
        source_info = st.session_state.ensemble_retriever.get_relevant_documents(prompt)
        if source_info and is_valid_url(source_info[0].metadata['source']):
            formatted_output += f"<p>Source: <a href='{source_info[0].metadata['source']}' target='_blank'>출처</a></p>"
        if source_info and is_valid_url(source_info[1].metadata['source']):
            formatted_output += f"<p>Source: <a href='{source_info[1].metadata['source']}' target='_blank'>출처</a></p>"
        # formatted_output += f"<p>Source: <a href='{result.metadata['source']}' target='_blank'>출처</a></p>"

        # with st.container():
        #     message_placeholder = st.empty()  # 결과를 표시할 스트림릿 컨테이너 준비
        display_gradually(formatted_output, message_placeholder)  # 점진적 표시 함수 호출


    st.session_state.messages.append({"role": "assistant", "content": formatted_output})



