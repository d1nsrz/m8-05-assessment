"""
CodeClarify — Streamlit chat UI.

Run with:
    streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from llm_service import ChatService

st.set_page_config(page_title="CodeClarify", page_icon="")
st.title("CodeClarify")
st.caption("Paste code and I'll explain it, step by step.")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.5,
        value=0.4, step=0.1,
        help="Lower = more precise explanations. Higher = more creative prose.",
    )
    model = st.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-1.5-flash"],
        index=0,
        help="gemini-2.0-flash is faster; gemini-1.5-flash may be more verbose.",
    )
    if st.button("Clear chat"):
        st.session_state.pop("service", None)
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- Paste any snippet — Python, JS, SQL, Bash …\n"
        "- Ask follow-ups about the same code\n"
        "- I'll flag bugs and gotchas automatically"
    )

if "service" not in st.session_state:
    st.session_state.service = ChatService(model=model, temperature=temperature)
if "messages" not in st.session_state:
    st.session_state.messages = []

service: ChatService = st.session_state.service
service.temperature = temperature

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Paste some code or ask a question about it…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply = st.write_stream(service.stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.divider()
    st.caption(
        f"**Token usage**  \n"
        f"↑ in: {service.total_input_tokens:,}  \n"
        f"↓ out: {service.total_output_tokens:,}"
    )
