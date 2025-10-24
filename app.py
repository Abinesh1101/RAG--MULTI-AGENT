"""
Streamlit Web UI for Multi-Agent RAG System
============================================
Interactive web interface for the Vision-Enhanced AI Assistant
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.controller_agent import ControllerAgent

# Page configuration
st.set_page_config(
    page_title="Multi-Agent RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .fact-badge { background-color: #4CAF50; color: white; }
    .analysis-badge { background-color: #2196F3; color: white; }
    .summary-badge { background-color: #FF9800; color: white; }
    .visual-badge { background-color: #9C27B0; color: white; }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_system():
    """Initialize the multi-agent system"""
    try:
        with st.spinner("🚀 Initializing Multi-Agent System... (1-2 minutes)"):
            st.session_state.controller = ControllerAgent(vector_store_path="data")
            st.session_state.initialized = True
        st.success("✅ System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return False


def get_intent_badge(intent):
    """Return HTML badge for intent"""
    badges = {
        'fact': '<span class="intent-badge fact-badge">📊 FACT</span>',
        'analysis': '<span class="intent-badge analysis-badge">🔍 ANALYSIS</span>',
        'summary': '<span class="intent-badge summary-badge">📝 SUMMARY</span>',
        'visual': '<span class="intent-badge visual-badge">🖼️ VISUAL</span>'
    }
    return badges.get(intent, f'<span class="intent-badge">{intent.upper()}</span>')


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Agent RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Vision-Enhanced, Intent-Aware AI Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system combines:
        - 🎯 **Intent Classification** (ML)
        - 📚 **Multimodal RAG** (Text + Images)
        - 👁️ **Vision Understanding** (BLIP-2)
        - 🧠 **LLM Reasoning** (Mistral 7B)
        - 🤝 **Multi-Agent Orchestration**
        """)
        
        st.divider()
        
        st.header("📖 Example Queries")
        st.write("""
        **Fact:**
        - How many vehicles did Tesla deliver?
        
        **Analysis:**
        - Compare Tesla and Google initiatives
        
        **Summary:**
        - Summarize key achievements
        
        **Visual:**
        - What does the emissions chart show?
        """)
        
        st.divider()
        
        st.header("📊 System Status")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            st.info(f"💬 Queries processed: {len(st.session_state.history)}")
        else:
            st.warning("⚠️ System not initialized")
        
        st.divider()
        
        if st.button("🔄 Reset System"):
            st.session_state.controller = None
            st.session_state.history = []
            st.session_state.initialized = False
            st.rerun()
    
    # Main content
    if not st.session_state.initialized:
        st.info("👋 Welcome! Click the button below to initialize the system.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize System", use_container_width=True):
                initialize_system()
                st.rerun()
    
    else:
        # Query input
        st.header("💬 Ask a Question")
        
        query = st.text_input(
            "Your question:",
            placeholder="e.g., How many vehicles did Tesla deliver in 2023?",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            process_button = st.button("🔍 Process Query", use_container_width=True)
        
        # Process query
        if process_button and query:
            with st.spinner("⏳ Processing your query... (2-5 minutes on CPU)"):
                try:
                    # Process query
                    result = st.session_state.controller.process_query(query)
                    
                    # Add to history
                    st.session_state.history.insert(0, result)
                    
                    # Display result
                    st.divider()
                    st.header("✅ Result")
                    
                    # Intent badge
                    st.markdown(get_intent_badge(result['intent']), unsafe_allow_html=True)
                    
                    # Answer
                    st.markdown("### 💡 Answer")
                    st.write(result['answer'])
                    
                    # Retrieved documents
                    if result.get('retrieved_docs'):
                        with st.expander(f"📚 Retrieved {len(result['retrieved_docs'])} documents"):
                            for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                                if isinstance(doc, dict):
                                    st.markdown(f"**Document {i}:** {doc.get('source', 'Unknown')}")
                                    st.caption(doc.get('text', '')[:200] + "...")
                                    st.divider()
                    
                    # Images
                    if result.get('image_descriptions'):
                        with st.expander(f"🖼️ Analyzed {len(result['image_descriptions'])} images"):
                            for img in result['image_descriptions']:
                                st.markdown(f"**{img['image_name']}**")
                                st.caption(img['description'])
                                st.divider()
                    
                    st.success("✅ Query processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        
        # Query history
        if st.session_state.history:
            st.divider()
            st.header("📜 Query History")
            
            for i, result in enumerate(st.session_state.history[:5], 1):
                with st.expander(f"Query {i}: {result['query'][:50]}..."):
                    st.markdown(get_intent_badge(result['intent']), unsafe_allow_html=True)
                    st.write(result['answer'][:300] + "...")


if __name__ == "__main__":
    main()
