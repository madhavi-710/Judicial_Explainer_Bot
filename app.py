import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Judicial Explainer Bot",
    page_icon="⚖️",
    layout="centered"
)

# ---------------- CUSTOM STYLING ----------------

st.title("⚖️ Judicial Explainer Bot")

st.markdown(
    "### Understand legal concepts in simple language"
)

st.write(
    "Ask questions about laws, legal terminology, "
    "the Indian judicial system, and legal procedures."
)

st.divider()

# ---------------- INITIALIZE AI MODEL ----------------


@st.cache_resource
def get_model():
    return ChatOllama(
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434",
        temperature=0.2,
        num_predict=250,
        num_ctx=2048
    )

llm = get_model()

# ---------------- SYSTEM INSTRUCTIONS ----------------


system_prompt = """
You are Judicial Explainer Bot, an educational assistant focused on Indian law.

Explain legal concepts and procedures in simple English.
Answer directly using short paragraphs or bullet points.
Give a simple example when useful.
Do not invent laws, legal sections, court judgments, or citations.
If uncertain, say so.
Provide general legal information, not personalized legal advice.
Recommend consulting a qualified lawyer for important legal decisions.
"""


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("About this chatbot")

    st.write(
        "This chatbot helps users understand legal concepts "
        "and judicial procedures in simple language."
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Educational tool only. Responses may contain errors "
        "and should not replace advice from a qualified lawyer."
    )

# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------

user_input = st.chat_input(
    "Ask a legal question..."
)

if user_input:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare conversation for LangChain
    chat_messages = [
        SystemMessage(content=system_prompt)
    ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_messages.append(
                HumanMessage(content=message["content"])
            )
        else:
            chat_messages.append(
                AIMessage(content=message["content"])
            )

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Analyzing your question..."):

            try:
                response = llm.invoke(chat_messages)

                answer = response.content

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(
                    "Unable to generate a response. "
                    "Please check whether Ollama is running."
                )

                st.caption(str(e))

                # Remove the failed user message
                st.session_state.messages.pop()