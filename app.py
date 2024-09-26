# Import dependencies
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  # Import Document class

# Set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = api_key

# Define constants
namespace = "wondervector5000"
model_name = 'text-embedding-3-small'

index_name = "memorymultidocs"
chunk_size = 1000

USERNAME = "User"
PASSWORD = "Password123"

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
docsearch = PineconeVectorStore.from_documents(
    documents="", 
    index_name=index_name, 
    embedding=embeddings, 
    namespace=namespace
)
time.sleep(1)

# Set up LLM and QA chain
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini', temperature=0.1)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={"k": 10})
)

# Function to add background color
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #a0bfb9;
            color: #895051;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )


# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")
add_bg_from_url()

# Session state for login, feedback, question, and visibility
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False  # Control visibility of feedback box

# Login logic
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app logic (after login)
else:
    st.title("Envoy")


    # Show file upload section only if the database is empty
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file:
        if st.button("Submit the file"):
            with st.spinner("Uploading and processing document..."):
                with open("uploaded_pdf.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("uploaded_pdf.pdf")
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
                documents = text_splitter.split_documents(pages)
                docsearch = PineconeVectorStore.from_documents(
                    documents=documents,
                    index_name=index_name,
                    embedding=embeddings,
                    namespace=namespace,
                )
            st.success("Document uploaded and processed. You can now ask questions about its content.")


    # Question input and response
    question = st.text_input("Ask queries related to the uploaded knowledge:")
    if st.button("Submit query"):
        with st.spinner("Getting your answer..."):
            retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            answer = qa.invoke(question)
            st.session_state.answer = answer["result"]
            st.session_state.question = question
            st.session_state.feedback_given = False
            st.session_state.feedback_submitted = False
            st.session_state.show_feedback_box = False  # Reset feedback box visibility

    # Display answer if question was submitted
    if st.session_state.question:
        st.write("Answer:", st.session_state.answer)

        # Button to reveal feedback box
        if not st.session_state.show_feedback_box:
            if st.button("Give Feedback"):
                st.warning("Only submit feedback if it is too much necessary. Giving wrong or too many feedback may make the model confuse!!")
                st.session_state.show_feedback_box = True

        # Feedback section (only show if "Give Feedback" is pressed)
        if st.session_state.show_feedback_box and not st.session_state.feedback_given:
            st.session_state.feedback_text = st.text_input("Write briefly about the problem with the response")
            if st.button("Submit Feedback"):
                st.session_state.feedback_given = True
                st.session_state.feedback_submitted = True

        # Display feedback if submitted
        if st.session_state.feedback_submitted:
            st.write("### Feedback Summary:")
            st.write(f"**Question**: {st.session_state.question}")
            st.write(f"**Answer**: {st.session_state.answer}")
            st.write(f"**Feedback**: {st.session_state.feedback_text}")
            
            memory_reinforcement = (
                "Hello LLM, we've identified a recent interaction where your performance did not meet expectations. "
                f"The question posed was: '{st.session_state.question}'. "
                f"Unfortunately, your response was: '{st.session_state.answer}'. "
                f"The correct response should have been: '{st.session_state.feedback_text}'. "
                "Please use this information to improve future responses."
            )      
            
            # Convert raw text to Document object
            document_reinf = Document(page_content=memory_reinforcement)
            
            # Store the document in Pinecone
            docsearch = PineconeVectorStore.from_documents(
                documents=[document_reinf],  # Pass the document inside a list
                index_name=index_name,
                embedding=embeddings,
                namespace=namespace,
            )
            
            st.success("Model memory updated")


    # Clear database button
    if st.button("Clear the database"):
        with st.spinner("Clearing the database..."):
            try:
                pc = Pinecone(api_key=api_key)
                index = pc.Index(index_name)
                index.delete(delete_all=True, namespace=namespace)
                st.success("Database cleared!")
            except:
                st.error("The database is already empty.")
