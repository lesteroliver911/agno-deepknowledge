import streamlit as st
from textwrap import dedent
import os
import shutil
from typing import Optional
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.vectordb.lancedb import LanceDb, SearchType
import time

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

def initialize_knowledge_base(uploaded_file=None):
    """Initialize the knowledge base with uploaded file only"""
    try:
        # Clear existing vector database and text files
        if os.path.exists("tmp/lancedb"):
            shutil.rmtree("tmp/lancedb")
            time.sleep(0.5)
        if os.path.exists("data/txt_files"):
            shutil.rmtree("data/txt_files")
            time.sleep(0.5)
        
        # Create fresh directories
        os.makedirs("tmp/lancedb", exist_ok=True)
        os.makedirs("data/txt_files", exist_ok=True)
        
        # Check if file is uploaded
        if uploaded_file is None:
            st.sidebar.warning("Please upload a text file to begin.")
            return None
            
        # Process uploaded file
        try:
            file_contents = uploaded_file.getvalue().decode("utf-8")
            upload_path = os.path.join("data/txt_files", uploaded_file.name)
            with open(upload_path, "w") as f:
                f.write(file_contents)
            st.sidebar.success(f"File uploaded: {uploaded_file.name}")
            
            # Initialize knowledge base with only the uploaded file
            agent_knowledge = TextKnowledgeBase(
                path="data/txt_files",
                vector_db=LanceDb(
                    uri="tmp/lancedb",
                    table_name="deep_knowledge_knowledge",
                    search_type=SearchType.hybrid,
                    embedder=OpenAIEmbedder(id="text-embedding-ada-002"),
                ),
            )
            
            agent_knowledge.load()
            return agent_knowledge
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error initializing knowledge base: {str(e)}")
        return None

def get_agent_storage():
    """Return agent storage"""
    return SqliteAgentStorage(
        table_name="deep_knowledge_sessions", 
        db_file="tmp/agents.db"
    )

def create_agent(session_id: Optional[str] = None, uploaded_file=None) -> Optional[Agent]:
    """Create and return a configured DeepKnowledge agent."""
    try:
        agent_knowledge = initialize_knowledge_base(uploaded_file)
        if agent_knowledge is None:
            return None
            
        agent_storage = get_agent_storage()
        
        return Agent(
            name="DeepKnowledge",
            session_id=session_id,
            model=OpenAIChat(
                id="gpt-4o",
                temperature=0,
            ),
            description=dedent("""\
            You are DeepKnowledge, an advanced reasoning agent designed to provide thorough,
            well-researched answers to any query by searching your knowledge base."""),
            instructions=dedent("""\
            Your mission is to leave no stone unturned in your pursuit of the correct answer.
            To achieve this, follow these steps:
            1. Analyze the input and break it down into key components.
            2. Search terms: Identify at least 3-5 key search terms.
            3. Initial Search: Search your knowledge base making at least 3 searches.
            4. Evaluation and synthesis of information.
            5. Clear documentation of reasoning and sources.
            
            Important: Only provide the final synthesized answer. Do not show your search process or intermediate steps."""),
            knowledge=agent_knowledge,
            storage=agent_storage,
            add_history_to_messages=True,
            num_history_responses=3,
            show_tool_calls=False,
            read_chat_history=True,
            markdown=True,
        )
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def get_example_topics():
    """Return a list of example topics"""
    return [
        "What are AI agents and how do they work in Agno?",
        "What chunking strategies does Agno support for text processing?",
        "How can I implement custom tools in Agno?",
        "How does knowledge retrieval work in Agno?",
        "What types of embeddings does Agno support?",
    ]

def handle_session_selection():
    """Handle session selection in Streamlit"""
    agent_storage = get_agent_storage()
    existing_sessions = agent_storage.get_all_session_ids()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_session = st.button("New Session", use_container_width=True)
    
    with col2:
        if existing_sessions:
            load_session = st.button("Load Session", use_container_width=True)
        else:
            load_session = False
            
    if new_session:
        st.session_state.current_session_id = None
        st.session_state.agent = create_agent()
        st.session_state.chat_history = []
        st.rerun()
        
    if load_session and existing_sessions:
        selected_session = st.sidebar.selectbox(
            "Select a session:",
            existing_sessions,
            index=0
        )
        if selected_session:
            st.session_state.current_session_id = selected_session
            st.session_state.agent = create_agent(selected_session)
            st.session_state.chat_history = []  # Reset chat history
            st.rerun()

def main():
    st.set_page_config(
        page_title="Agno - DeepKnowledge Assistant",
        page_icon="🤔",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton button {
            font-size: 0.8rem;
            padding: 0.3rem 1rem;
        }
        .chat-message {
            font-size: 0.9rem;
            margin: 0.5rem 0;
            padding: 0.5rem;
        }
        .sidebar-title {
            font-size: 1.2rem !important;
            margin-bottom: 0.5rem !important;
        }
        .status-section {
            font-size: 0.9rem;
            margin: 1rem 0;
            padding: 0.5rem;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - All controls and status information
    with st.sidebar:
        st.markdown("<h1 class='sidebar-title'>Agno - DeepKnowledge Assistant</h1>", unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### 📁 Upload Knowledge")
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt"],
            help="Upload a text file to be used as the knowledge base"
        )
        
        # Status Section
        st.markdown("<div class='status-section'>", unsafe_allow_html=True)
        st.markdown("### System Status")
        if uploaded_file:
            st.markdown("✅ Agent Ready")
            st.markdown(f"📚 Knowledge Base: {uploaded_file.name}")
            if st.button("Process File", type="primary"):
                st.session_state.chat_history = []
                st.session_state.agent = None  # Force agent reinitialization
                st.rerun()
        else:
            st.markdown("⏳ Agent: Waiting for file")
            st.markdown("📚 Knowledge Base: No file uploaded")
            
        if st.session_state.current_session_id:
            st.markdown(f"🔑 Session: Active")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Controls Section (only show if file is uploaded)
        if uploaded_file:
            st.markdown("### 🛠️ Controls")
            handle_session_selection()
            
            if st.session_state.current_session_id:
                st.success(f"Session ID: {st.session_state.current_session_id}")
            
            if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main Area - Chat Only
    if not uploaded_file:
        st.info("👋 Welcome! Please upload a text file in the sidebar to start chatting.")
        return
    
    # Initialize agent if needed and file is uploaded
    if st.session_state.agent is None and uploaded_file is not None:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = create_agent(uploaded_file=uploaded_file)
            if st.session_state.agent is None:
                st.error("Failed to initialize agent. Please check your configuration and try again.")
                return
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message" style="background-color: #f0f2f6; border-radius: 5px;">
                    <strong>You:</strong> {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            content = message['content']
            if hasattr(content, 'content'):
                content = content.content
            st.markdown(f"""
                <div class="chat-message" style="background-color: #e8f0fe; border-radius: 5px;">
                    <strong>Assistant:</strong> {content}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input at bottom
    user_input = st.chat_input("Ask your question here...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response = st.session_state.agent.run(user_input)
            if hasattr(response, 'content'):
                response = response.content
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

if __name__ == "__main__":
    main()
