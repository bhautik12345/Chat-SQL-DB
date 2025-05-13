import streamlit as st
from pathlib import Path
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate
from sqlalchemy import create_engine
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------- 🔧 CONFIG --------------------
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🗃️ Chat with Your SQL Database")

# -------------------- ⚙️ DB SELECTION --------------------
LOCAL_DB = 'USE_LOCALDB'
MYSQL = 'USE_MYSQL'

st.sidebar.title('⚙️ Settings')
radio_options = ['Use SQLite3 (student.db)', 'Connect to MySQL']
selected_option = st.sidebar.radio('Choose a database to chat with:', radio_options)

# User inputs
if radio_options.index(selected_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input('MySQL Host')
    mysql_user = st.sidebar.text_input('MySQL Username')
    mysql_password = st.sidebar.text_input('MySQL Password', type='password')
    mysql_db = st.sidebar.text_input('MySQL Database Name')
else:
    db_uri = LOCAL_DB

google_api_key = st.sidebar.text_input('🔑 Enter your Google API key:', type='password')
os.environ['GOOGLE_API_KEY'] = google_api_key


# -------------------- 🛑 Validation --------------------
if not google_api_key:
    st.warning("⚠️ Please provide your Google API key.")
    st.stop()
if db_uri == MYSQL and not (mysql_host and mysql_user and mysql_password and mysql_db):
    st.warning("⚠️ Please complete MySQL connection details.")
    st.stop()

# -------------------- 🤖 LLM SETUP --------------------
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# -------------------- 🗄️ DB CONFIG --------------------
@st.cache_resource(ttl="2h")
def configure_db(db_type, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_type == LOCAL_DB:
        db_path = (Path(__file__).parent / "student.db").absolute()
        # st.info(f"📁 Using local SQLite3 DB:{db_path}")
        creator = lambda: sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        return SQLDatabase(create_engine('sqlite://', creator=creator))

    elif db_type == MYSQL:
        engine_url = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:3306/{mysql_db}"
        return SQLDatabase(create_engine(engine_url))

# Configure the selected DB
db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db) if db_uri == MYSQL else configure_db(db_uri)

# -------------------- 🧰 TOOLKIT & PROMPT --------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

prompt_template = PromptTemplate.from_template("""
You are a SQL expert assistant. You only answer questions that are strictly related to SQL, databases, or the connected database's content.You may engage in casual conversation with the user, but do not answer unrelated technical questions.

Use the following format for SQL/database questions:

Question: the input question you must answer  
Thought: you should always think about what to do  
Action: the action to take, should be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation can repeat N times)  
Thought: I now know the final answer  
Final Answer: the final answer to the original input question

You have access to the following tools:

{tools}

Begin!

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm=llm, tools=toolkit.get_tools(), prompt=prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="force"
)

# -------------------- 💬 CHAT UI --------------------
if 'messages' not in st.session_state or st.sidebar.button('🧹 Clear Chat History'):
    st.session_state['messages'] = [{'role': 'assistant', 'content': '👋 How can I help you today?'}]

# Display chat history
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# Get user input
user_prompt = st.chat_input("Write your SQL-related query here...")

if user_prompt:
    st.session_state['messages'].append({'role': 'user', 'content': user_prompt})
    st.chat_message('user').write(user_prompt)

    with st.chat_message('assistant'):
        with st.spinner("🧠 Thinking..."):
            try:
                response = agent_executor.invoke({'input': user_prompt})
                st.session_state['messages'].append({'role': 'assistant', 'content': response})
                st.write(response)
            except Exception as e:
                st.error(f"❌ Error: {e}")
