import streamlit as st
from streamlit_chat import message
from caller import Caller
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

st.title("Ferflexcity")
@st.cache_resource(show_spinner=True)
def create_caller():
    caller = Caller()
    return caller

search_bot = create_caller()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    bot_output, urls, titles = search_bot.search(prompt)

    combined_output = bot_output
    if urls and titles:
        combined_output += "\n\n### Helpful links:\n"
        for title, url in zip(titles, urls):
            combined_output += f"- [{title}]({url})\n"

    with st.chat_message("bot"):
        st.markdown(combined_output)  # Display the combined message

    st.session_state.messages.append({"role": "bot", "content": combined_output})  # Save the combined message
