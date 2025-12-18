from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.agents import create_agent
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """You are an expert software developer, Act as an expert full-stack engineer assistant specialized in Python, Django, Django REST Framework, React, and JavaScript. Your primary role is to provide high-quality code reviews, debugging help, architecture guidance, solve problems [for DSA, machine learning, data science, engineering level mathematics], and advice on best practices for modern web development.
Your name is Khosann GenAI, made and trained my Sanju Thapa.

Since you are now running on local server and I am currently your owner. My name is Sanju Thapa. I am Software Engineer and working as freelancer on machine learning and SaaS projects.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

class GeminiConfiguration:

    def __init__(self, api_key):
        # self.api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    def get_generation_config(self):
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048
        }
    
    def get_safety_settings(self):
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    def create_model(self, model_name):
        llm = create_agent(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver())
        return llm

st.title("🦜🔗 :rainbow[Khosann GenAI]")
if st.button("🧹 New Conversation"):
    st.session_state.messages = []
    st.rerun()
API_KEY = st.text_input("Enter your API KEY")
if __name__ == "__main__":
    try:
        config = GeminiConfiguration(api_key=API_KEY)
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key = config.api_key,
        )
        agent = config.create_model(model_name=model)

        for msg in st.session_state.messages:
          with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        text = st.chat_input("Ask your question...")

        # store user message
        if text:
          title = "Khosann Gen AI"
          with st.spinner(f"Searching :rainbow[{title}] data...", show_time=True):
              # st.write(text["files"][0])
              # st.markdown(f'**User:** :rainbow[{text}]')
              response = agent.invoke(
                {"messages": [{"role": "user", "content": f"{text}"}]},
                {"configurable": {"thread_id": "1", "checkpoint_ns": "chat"}},
              )
              ai_message = response['messages'][1]
              ai_content = (
                  ai_message.content
                  if hasattr(ai_message, 'content')
                  else ai_message['content']
              )
              st.session_state.messages.append({
                  "role": "user",
                  "content": text
              })
              st.session_state.messages.append({
                "role": "assistant",
                "content": ai_content
              })

              with st.chat_message("assistant"):
                  st.write(ai_content)
              
              st.toast("Congrats! your data has arrived", icon="🎉")
              st.balloons()
        # print(f"\nResponse\n{response.content}")
    except Exception as e:
        print(e)
