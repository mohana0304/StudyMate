import os
import streamlit as st
from backend import prepare_documents, create_faiss_index, save_faiss, load_faiss, search
from llm import generate_answer
import shutil

# Initialize session state for chat history and index (before any st.* calls)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None

# set_page_config MUST be the FIRST Streamlit command
st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")

# Custom CSS for medium font sizes (immediately after set_page_config)
st.markdown("""
    <style>
    .answer-text {
        font-size: 16px !important;  /* Medium size for answers */
        line-height: 1.5;
        color: #FFFFFF;
    }
    .snippet-text {
        font-size: 18px !important;  /* Slightly smaller for snippets */
        line-height: 1.4;
        color: #FFFFFF;
    }
    .chat-message {
        font-size: 16px !important;  /* Medium size for user messages */
        line-height: 1.5;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 StudyMate — Chat with Your PDFs")

# Ensure data directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/faiss_index", exist_ok=True)

# Sidebar
with st.sidebar:
    st.header("Setup / Notes")
    st.write("""
    - Upload one or more PDF files.
    - Click **Process PDFs** to extract & index.
    - Once indexed, use the chat input below to ask questions (press Enter to submit).
    - View the full conversation history above the input.
    - If no LLM is configured, the app returns top snippets as the answer.
    """)
    if st.button("Clear uploaded data"):
        shutil.rmtree("data/uploads", ignore_errors=True)
        shutil.rmtree("data/faiss_index", ignore_errors=True)
        os.makedirs("data/uploads", exist_ok=True)
        # Clear chat history and index
        st.session_state.messages = []
        st.session_state.index = None
        st.session_state.metadata = None
        st.experimental_rerun()

    # Display number of messages and indexed chunks
    st.markdown(f"**Chat messages:** {len(st.session_state.messages)}")
    st.markdown(f"**Indexed chunks:** {len(st.session_state.metadata) if st.session_state.metadata else 0}")
    # Slider for top_k
    top_k = st.slider("Number of retrieved chunks (k)", 1, 6, 3, key="top_k")

# File uploader
uploaded = st.file_uploader("Upload PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)

if uploaded:
    st.info(f"{len(uploaded)} file(s) uploaded. Click 'Process PDFs' to extract and index.")
    if st.button("Process PDFs"):
        saved_paths = []
        for f in uploaded:
            path = os.path.join("data/uploads", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(path)

        with st.spinner("Extracting and chunking PDFs..."):
            chunks = prepare_documents(saved_paths)
        st.success(f"Extracted {len(chunks)} text chunks from uploaded PDFs.")

        # Build and save index
        with st.spinner("Creating embeddings and FAISS index (this may take a bit)..."):
            index, metadata = create_faiss_index(chunks)
            save_faiss(index, metadata, dirpath="data/faiss_index")
            # Store in session state
            st.session_state.index = index
            st.session_state.metadata = metadata
        st.success("FAISS index created and saved. Start chatting below!")

# Load index if not already in session state
if st.session_state.index is None:
    index, metadata = load_faiss(dirpath="data/faiss_index")
    if index is not None:
        st.session_state.index = index
        st.session_state.metadata = metadata

# Chat interface
if st.session_state.index is None:
    st.warning("No indexed documents found. Upload and process PDFs to start chatting.")
else:
    st.success("Index loaded. Chat with your documents below!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)
            # Show snippets for assistant messages
            if message["role"] == "assistant" and "snippets" in message:
                st.markdown('<div class="snippet-text"><b>Retrieved Snippets (Sources)</b></div>', unsafe_allow_html=True)
                for i, r in enumerate(message["snippets"], start=1):
                    st.markdown(f'<div class="snippet-text"><b>[{i}] {r["doc_id"]} — page {r["page"]}]</b></div>', unsafe_allow_html=True)
                    snippet = r["text"]
                    st.markdown(f'<div class="snippet-text">{snippet[:1200] + ("..." if len(snippet) > 1200 else "")}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="snippet-text">---</div>', unsafe_allow_html=True)

    # Chat input (always visible when index is loaded)
    if query := st.chat_input("Ask a question about your documents:"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message">{query}</div>', unsafe_allow_html=True)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                results = search(st.session_state.index, st.session_state.metadata, query, top_k=top_k)
                if not results:
                    answer_content = '<div class="answer-text">No relevant content found in indexed documents.</div>'
                    snippets = []
                else:
                    try:
                        answer = generate_answer(query, results, max_tokens=250)
                        # Format answer with snippets
                        answer_content = f'<div class="answer-text"><b>Answer</b>: {answer}</div>\n\n<div class="snippet-text"><b>Retrieved Snippets (Sources)</b></div>\n'
                        for i, r in enumerate(results, start=1):
                            snippet = r["text"]
                            answer_content += f'<div class="snippet-text"><b>[{i}] {r["doc_id"]} — page {r["page"]}]</b><br>{snippet[:1200] + ("..." if len(snippet) > 1200 else "")}<br>---</div>\n'
                        snippets = results
                    except Exception as e:
                        answer_content = f'<div class="answer-text">Error generating answer: {str(e)}. Showing retrieved snippets instead.</div>\n\n<div class="snippet-text"><b>Retrieved Snippets (Sources)</b></div>\n'
                        for i, r in enumerate(results, start=1):
                            snippet = r["text"]
                            answer_content += f'<div class="snippet-text"><b>[{i}] {r["doc_id"]} — page {r["page"]}]</b><br>{snippet[:1200] + ("..." if len(snippet) > 1200 else "")}<br>---</div>\n'
                        snippets = results

            # Display assistant response
            st.markdown(answer_content, unsafe_allow_html=True)

            # Append to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_content,
                "snippets": snippets
            })

# import os
# import streamlit as st
# from backend import prepare_documents, create_faiss_index, save_faiss, load_faiss, search
# from llm import generate_answer
# import shutil
# import json
# import time

# # ==============================
# # SESSION STATE INITIALIZATION
# # ==============================
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "index" not in st.session_state:
#     st.session_state.index = None
# if "metadata" not in st.session_state:
#     st.session_state.metadata = None
# if "theme" not in st.session_state:
#     st.session_state.theme = "dark"

# # ==============================
# # PAGE CONFIG & CUSTOM STYLES
# # ==============================
# st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")

# # Light/Dark theme CSS
# if st.session_state.theme == "dark":
#     bg_color = "#1E1E1E"
#     text_color = "#FFFFFF"
# else:
#     bg_color = "#FFFFFF"
#     text_color = "#000000"

# st.markdown(
#     f"""
#     <style>
#     body {{
#         background-color: {bg_color};
#         color: {text_color};
#     }}
#     .answer-text {{
#         font-size: 16px !important;
#         line-height: 1.5;
#         color: {text_color};
#     }}
#     .snippet-text {{
#         font-size: 15px !important;
#         line-height: 1.4;
#         color: {text_color};
#     }}
#     .chat-message {{
#         font-size: 16px !important;
#         line-height: 1.5;
#         margin-bottom: 10px;
#         color: {text_color};
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("📚 StudyMate — Chat with Your PDFs")

# # ==============================
# # DATA DIRECTORIES
# # ==============================
# os.makedirs("data/uploads", exist_ok=True)
# os.makedirs("data/faiss_index", exist_ok=True)

# # ==============================
# # SIDEBAR
# # ==============================
# with st.sidebar:
#     st.header("⚙️ Settings & Actions")

#     # Theme toggle
#     if st.button("🌗 Toggle Theme"):
#         st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
#         st.experimental_rerun()

#     # Clear uploaded data
#     if st.button("🗑️ Clear uploaded data"):
#         shutil.rmtree("data/uploads", ignore_errors=True)
#         shutil.rmtree("data/faiss_index", ignore_errors=True)
#         os.makedirs("data/uploads", exist_ok=True)
#         st.session_state.messages = []
#         st.session_state.index = None
#         st.session_state.metadata = None
#         st.experimental_rerun()

#     # Clear chat only
#     if st.button("🧹 Clear chat history"):
#         st.session_state.messages = []
#         st.experimental_rerun()

#     # Download chat
#     if st.session_state.messages:
#         chat_json = json.dumps(st.session_state.messages, indent=2)
#         st.download_button("💾 Download Chat", chat_json, "chat_history.json")

#     st.markdown(f"**Chat messages:** {len(st.session_state.messages)}")
#     st.markdown(f"**Indexed chunks:** {len(st.session_state.metadata) if st.session_state.metadata else 0}")

#     # Retrieval top_k
#     top_k = st.slider("🔍 Retrieved chunks (k)", 1, 6, 3, key="top_k")

# # ==============================
# # FILE UPLOADER & INDEXING
# # ==============================
# uploaded = st.file_uploader("Upload PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)

# if uploaded:
#     st.info(f"{len(uploaded)} file(s) uploaded. Click 'Process PDFs' to extract and index.")
#     if st.button("📑 Process PDFs"):
#         saved_paths = []
#         for f in uploaded:
#             path = os.path.join("data/uploads", f.name)
#             with open(path, "wb") as out:
#                 out.write(f.getbuffer())
#             saved_paths.append(path)

#         with st.spinner("📖 Extracting and chunking PDFs..."):
#             chunks = prepare_documents(saved_paths)
#         st.success(f"✅ Extracted {len(chunks)} text chunks from uploaded PDFs.")

#         with st.spinner("⚡ Creating embeddings and FAISS index..."):
#             index, metadata = create_faiss_index(chunks)
#             save_faiss(index, metadata, dirpath="data/faiss_index")
#             st.session_state.index = index
#             st.session_state.metadata = metadata
#         st.success("🎉 FAISS index created. Start chatting below!")

# # Load saved FAISS index if available
# if st.session_state.index is None:
#     index, metadata = load_faiss(dirpath="data/faiss_index")
#     if index is not None:
#         st.session_state.index = index
#         st.session_state.metadata = metadata

# # ==============================
# # CHAT INTERFACE
# # ==============================
# if st.session_state.index is None:
#     st.warning("⚠️ No indexed documents found. Upload and process PDFs to start chatting.")
# else:
#     st.success("✅ Index loaded. Chat with your documents below!")

#     # Display chat history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             # User message
#             if message["role"] == "user":
#                 st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)

#             # Assistant message
#             elif message["role"] == "assistant":
#                 # Force white color for answers
#                 st.markdown(
#                     f'<div class="answer-text" style="color:#FFFFFF;">{message["content"]}</div>',
#                     unsafe_allow_html=True,
#                 )

#                 # Show snippets (sources)
#                 if "snippets" in message:
#                     st.markdown('<div class="snippet-text"><b>Retrieved Snippets (Sources)</b></div>', unsafe_allow_html=True)
#                     for j, r in enumerate(message["snippets"], start=1):
#                         snippet = r["text"]
#                         st.markdown(
#                             f'<div class="snippet-text" style="color:#CCCCCC;"><b>[{j}] {r["doc_id"]} — page {r["page"]}</b><br>'
#                             f'{snippet[:800] + ("..." if len(snippet) > 800 else "")}<br>---</div>',
#                             unsafe_allow_html=True,
#                         )

#                 # Regenerate button for last assistant answer
#                 if i == len(st.session_state.messages) - 1:
#                     if st.button("🔄 Regenerate Answer", key=f"regen_{i}"):
#                         query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else None
#                         if query:
#                             results = search(st.session_state.index, st.session_state.metadata, query, top_k=top_k)
#                             answer = generate_answer(query, results, max_tokens=250)
#                             st.session_state.messages[-1]["content"] = f"**Answer (Regenerated)**: {answer}"
#                             st.experimental_rerun()

#     # ==============================
#     # CHAT INPUT
#     # ==============================
#     if query := st.chat_input("💬 Ask a question about your documents:"):
#         st.session_state.messages.append({"role": "user", "content": query})

#         with st.chat_message("user"):
#             st.markdown(f'<div class="chat-message">{query}</div>', unsafe_allow_html=True)

#         with st.chat_message("assistant"):
#             with st.spinner("🤖 Thinking..."):
#                 results = search(st.session_state.index, st.session_state.metadata, query, top_k=top_k)
#                 if not results:
#                     answer = "No relevant content found."
#                     snippets = []
#                 else:
#                     try:
#                         answer = generate_answer(query, results, max_tokens=250)
#                         snippets = results
#                     except Exception as e:
#                         answer = f"⚠️ Error: {str(e)}. Showing retrieved snippets instead."
#                         snippets = results

#                 # Always white text for answers
#                 st.markdown(
#                     f'<div class="answer-text" style="color:#FFFFFF;"><b>Answer</b>: {answer}</div>',
#                     unsafe_allow_html=True,
#                 )

#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": f"<b>Answer</b>: {answer}", "snippets": snippets}
#                 )
