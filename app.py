import os
import streamlit as st
from fpdf import FPDF

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

from langchain_groq import ChatGroq


# ============================================================
# 🔐 HARD-CODED API KEY
# ============================================================
GROQ_API_KEY = "gsk_cvJBrBjH6XIdj1X"


# ============================================================
# SAFE TEXT EXTRACTION
# ============================================================
def extract_message_text(msg):
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)


# ============================================================
# GROQ MODEL CALL
# ============================================================
def call_groq(messages, model_name):
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name
        )
        response = llm.invoke(messages)

        text = response if isinstance(response, str) else extract_message_text(response)
        return [{"role": "assistant", "content": text}]

    except Exception as e:
        return [{"role": "assistant", "content": f"Groq Error: {e}"}]


# ============================================================
# LANGGRAPH
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(model_name):
    def inner(state: State):
        msgs = []
        for m in state["messages"]:
            if hasattr(m, "role") and hasattr(m, "content"):
                msgs.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                msgs.append({"role": m.get("role"), "content": m.get("content")})
            else:
                msgs.append({"role": "user", "content": str(m)})

        return {"messages": call_groq(msgs, model_name)}
    return inner


def build_graph(model_name):
    g = StateGraph(State)
    g.add_node("chatbot", chatbot_node(model_name))
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", END)
    return g.compile()


# ============================================================
# PDF EXPORT
# ============================================================
def export_chat_to_pdf(history, path="chat.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Chat Export", ln=True)

    for role, msg in history:
        pdf.multi_cell(0, 8, f"{role.upper()}: {msg}")

    pdf.output(path)
    return path


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="SmartFlow AI — Multi-Groq Models", layout="wide")
st.title("🤖 SmartFlow AI — Groq Models")


# Predefined Groq models
groq_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama-guard-3-8b",

    "meta-llama/llama-4-maverick-17b-128k",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b"
]



if "history" not in st.session_state:
    st.session_state.history = []


# Sidebar Model Selection
with st.sidebar:
    st.subheader("Groq Model Selection")

    selected_model = st.selectbox("Choose a model:", groq_models)

    custom_model = st.text_input("Or paste custom model name:", "")

    # If user enters a custom model, override
    model_name = custom_model if custom_model.strip() else selected_model

st.success(f"Using model: {model_name}")

graph = build_graph(model_name)


# ============================================================
# MAIN CHAT
# ============================================================
user_text = st.text_input("Enter your message:")

if st.button("Send"):
    final_state = graph.invoke({"messages": ({"role": "user", "content": user_text})})

    bot_msg = extract_message_text(final_state["messages"][-1])

    st.session_state.history.append(("you", user_text))
    st.session_state.history.append(("ai", bot_msg))


st.write("---")
st.subheader("📜 Chat History")

for role, msg in st.session_state.history:
    st.write(f"**{role.upper()}:** {msg}")


# Export PDF
if st.button("Export as PDF"):
    pdf_file = export_chat_to_pdf(st.session_state.history)
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name="chat.pdf")
