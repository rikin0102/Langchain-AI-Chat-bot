import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="LangChain QA Chatbot", page_icon="🤖")
st.title("Simple LangChain Chat with Groq")
st.markdown("working demo of LangChain integration with Groq's ultra-fast inference")

#SIDEBAR
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
        ],
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

#STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

#CHAIN
@st.cache_resource
def get_chain(api_key, model_name):
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.7,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq."),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()

if not api_key:
    st.warning("Please enter your Groq API key to start chatting.")
    st.stop()

chain = get_chain(api_key, model_name)

# CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# INPUT SECTION
if question := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({"question": question}):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
