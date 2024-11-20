import streamlit as st
import os
from rag_system import build_knowledge_base, setup_query_engine, rag

# Set page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session_state()
    
    # Header
    st.title("📚 Local RAG System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        st.markdown("Upload your PDF files to the `data/my_knowledge_base` directory.")
        
        if st.button("🔄 Rebuild Knowledge Base"):
            with st.spinner("Building knowledge base..."):
                # Create directory if it doesn't exist
                os.makedirs('data/my_knowledge_base/', exist_ok=True)
                # Build the knowledge base
                st.session_state.index = build_knowledge_base()
                if st.session_state.index:
                    st.session_state.query_engine = setup_query_engine(st.session_state.index)
                    st.success("Knowledge base rebuilt successfully!")
                else:
                    st.error("Failed to build knowledge base. Please check your documents.")

        st.markdown("---")
        st.markdown("### 📑 Available Documents")
        try:
            docs = os.listdir('data/my_knowledge_base/')
            if docs:
                for doc in docs:
                    if doc.lower().endswith('.pdf'):
                        st.text(f"📄 {doc}")
            else:
                st.info("No documents found. Please add PDFs to the knowledge base directory.")
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")

    # Main chat interface
    if st.session_state.query_engine is None:
        st.info("Please rebuild the knowledge base to start querying.")
    else:
        # Chat interface
        st.markdown("### 💬 Chat Interface")
        
        # Display chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
            st.markdown("---")

        # Query input
        query = st.text_input("Enter your question:", key="query_input", placeholder="Ask me anything about the documents...")
        
        if st.button("Send"):
            if query:
                with st.spinner("Generating response..."):
                    response = rag(query, st.session_state.query_engine)
                    st.session_state.chat_history.append((query, response))
                    st.experimental_rerun()
            else:
                st.warning("Please enter a question.")

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()
