# ✅ Fix missing asyncio event loop (Windows-only issue)
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ✅ Imports
import os
import json
import re
import tempfile
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

# ✅ Load API Keys
api_key = st.secrets["GOOGLE_API_KEY"]
tavily_key = st.secrets["TAVILY_API_KEY"]
os.environ["TAVILY_API_KEY"] = tavily_key

# ✅ Initialize Models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

# ✅ Prompt Templates
extract_prompt = PromptTemplate(
    input_variables=["doc_text"],
    template="""
You are a smart assistant. From the following document text, extract and return this information as a valid JSON object:
- name
- age
- gender
- income (in ₹)
- caste or category
- state
- occupation

Document:
{doc_text}

Respond only with JSON:
"""
)

eligibility_prompt = PromptTemplate(
    input_variables=["user_data", "scheme_text"],
    template="""
You are an eligibility checker.

Here is the user's info in JSON:
{user_data}

Here is a government scheme:
{scheme_text}

Decide if the user is eligible. Reply:
- YES (with a short reason), or
- NO (with reason)

Just respond in plain text.
"""
)

# ✅ Chains
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
eligibility_chain = LLMChain(llm=llm, prompt=eligibility_prompt)

# ✅ Function: PDF to Text
def extract_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n\n".join([p.page_content for p in pages])

# ✅ Function: Extract User Info
def extract_user_info(doc_text):
    result = extract_chain.run(doc_text=doc_text)
    cleaned = re.sub(r"```json|```", "", result).strip()
    try:
        return json.loads(cleaned)
    except:
        return {"error": "Invalid JSON", "raw_output": result}

# ✅ Function: Check Eligibility
def match_eligibility(user_data, scheme_text):
    return eligibility_chain.run(
        user_data=json.dumps(user_data),
        scheme_text=scheme_text
    )

# ✅ Build Vector DB if not exists
def build_vector_db_from_text():
    file_path = "govt_schemes.txt"  # ✅ Use full path if needed
    if not os.path.exists(file_path):
        raise FileNotFoundError("⚠️ govt_schemes.txt not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    splitter = CharacterTextSplitter(separator="\n---\n", chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(data)

    documents = [
        Document(page_content=chunk, metadata={"title": chunk.split("\n")[0].replace("Title: ", "")})
        for chunk in chunks
    ]

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local("govt_schemes_index")

# ✅ Load Vector DB
def load_vector_db():
    if not os.path.exists("govt_schemes_index/index.faiss"):
        build_vector_db_from_text()
    return FAISS.load_local("govt_schemes_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Streamlit UI Starts Here
st.set_page_config(page_title="Scheme Eligibility Checker", page_icon="📋")
st.title("🗞 Government Scheme Eligibility Checker")

uploaded_file = st.file_uploader("📤 Upload your PDF with personal info", type=["pdf"])

if uploaded_file and "user_data" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_path = temp_pdf.name

    st.success("✅ PDF uploaded successfully.")
    with st.spinner("📄 Extracting text..."):
        doc_text = extract_pdf_text(temp_path)
        st.session_state.doc_text = doc_text
        st.text_area("Extracted Text", doc_text[:800])

    user_data = extract_user_info(doc_text)
    st.session_state.user_data = user_data

    with st.spinner("🔍 Loading government schemes..."):
        vectordb = load_vector_db()
        st.session_state.vectordb = vectordb

    query = f"Schemes for a {user_data.get('age')} year old {user_data.get('caste or category')} from {user_data.get('state')}"
    results = vectordb.similarity_search(query, k=5)
    st.session_state.results = results

    st.subheader("📋 Eligibility Results")
    eligible_count = 0
    for i, r in enumerate(results, start=1):
        title = r.metadata.get("title", f"Scheme #{i}")
        eligibility = match_eligibility(user_data, r.page_content)

        st.markdown(f"### 🏛️ {title}")
        st.write(eligibility)

        if "YES" in eligibility.upper():
            eligible_count += 1

    st.success(f"🎉 You are eligible for {eligible_count} out of {len(results)} schemes.")

# ✅ Chat Assistant with Fallback to Tavily Search
if "user_data" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask About Eligible Schemes")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi! I'm your assistant. Ask me anything about the schemes you're eligible for.")
        ]

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("ai").write(msg.content)

    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.chat_message("user").write(user_query)

        memory = ConversationBufferMemory(return_messages=True)
        memory.chat_memory.messages = st.session_state.chat_history

        search_tool = TavilySearchResults(k=3)
        chat_history = memory.chat_memory.messages
        chat_history.append(HumanMessage(content=user_query))

        try:
            response = llm.invoke(chat_history)
            content = response.content

            # ✅ Fallback to Tavily Search if model doesn't know
            if "I don't know" in content or "not sure" in content.lower():
                search_results = search_tool.invoke({"query": user_query})
                fallback = "\n\n🔍 I searched the web:\n"
                for i, res in enumerate(search_results, 1):
                    fallback += f"{i}. [{res['title']}]({res['url']})\n"
                content += fallback

            reply = AIMessage(content=content)
            st.chat_message("ai").write(content)
            st.session_state.chat_history.append(reply)

        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.chat_message("ai").write(error_msg)
