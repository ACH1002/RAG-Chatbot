import dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from utils import load_data, refresh_data_on_demand, is_valid_url, format_response, display_gradually

dotenv.load_dotenv()

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    data = load_data()
    st.session_state._fs = LocalFileStore("./cache/")

    # 문서를 청크로 나눔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    # 임베딩 로드
    hf = HuggingFaceEmbeddings(model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr')
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(hf, st.session_state._fs, namespace="sentence-transformer")

    # FAISS 벡터 스토어 생성
    faiss_vector_store = FAISS.from_documents(documents=all_splits, embedding=cached_embedder)

    # 리트리버 생성
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5
    st.session_state.ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weight=[0.8, 0.2])

    # LLM 설정
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 리트리버 도구 생성
    tool = create_retriever_tool(st.session_state.ensemble_retriever, "INU-Notice-Service", "Searches and returns documents regarding the notice service guide")
    tools = [tool]

    # 메모리 설정
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    # 시스템 메시지 생성
    system_message = SystemMessage(content=(
        "You are an assistant for question-answering tasks." 
        "Answer only the questions based on the text provided."
        "Use the following pieces of retrieved context to answer the question." 
        "If you don't know the answer, just say that you don't know."
        "\nQuestion: {question} \nContext: {context} \nAnswer:"
        ))

    # 메모리와 시스템 메시지를 포함한 프롬프트 생성
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message, extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)])

    # 프롬프트를 사용하여 에이전트 생성
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    # 에이전트 실행기 생성
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False, return_intermediate_steps=True)
    st.session_state.agent_executor = agent_executor

st.title("AI 인천대학교 산업경영공학과 상담원")

if st.button("refresh data"):
    refresh_data_on_demand(_fs=st.session_state._fs, _agent=st.session_state.agent_executor.agent, _memory=st.session_state.agent_executor.memory)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        result = st.session_state.agent_executor({"input": prompt})
        output = result['output']
        formatted_output = format_response(output)

        # 출처 정보 추가
        source_info = st.session_state.ensemble_retriever.get_relevant_documents(prompt)
        if source_info and is_valid_url(source_info[0].metadata['source']):
            formatted_output += f"<p>출처: <a href='{source_info[0].metadata['source']}' target='_blank'>출처</a></p>"


        display_gradually(formatted_output, message_placeholder)

    st.session_state.messages.append({"role": "assistant", "content": formatted_output})
