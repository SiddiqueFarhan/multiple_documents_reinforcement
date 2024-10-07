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
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = api_key


uri = "mongodb+srv://nic:mydatabase@cluster0.0plxa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Access your database (replace 'applicationinfo' with your actual database name)
db = client['applicationinfo']

# Access collections for 'questions' and 'uploadedpdf'
questions_collection = db['questions_old']
uploadedpdf_collection = db['uploadedpdf_old']

# Functions for 'questions' collection
def insert_question(question, answer, feedback):
    """Inserts a question into the 'questions' collection."""
    data = {"question": question, "asnwer": answer, "feedback": feedback}
    result = questions_collection.insert_one(data)


# Functions for 'uploadedpdf' collection
def insert_pdf_data(pdf_data):
    """Inserts PDF data into the 'uploadedpdf' collection."""
    data = {"pdf_content": pdf_data}
    result = uploadedpdf_collection.insert_one(data)





# Define constants
namespace = "wondervector5000"
model_name = 'text-embedding-3-small'
index_name = "modelmemory"
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

# Function to check if the Pinecone index is empty
def is_index_empty():
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        # Query Pinecone to check the number of vectors in the index
        stats = index.describe_index_stats(namespace=namespace)
        vector_count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
        return vector_count == 0  # Return True if no vectors are stored
    except Exception as e:
        st.error(f"Error checking index status: {e}")
        return True

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

    # Check if Pinecone database is empty
    if is_index_empty():
        # Show file upload section only if the database is empty
        st.write("Upload documents")
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
                # Concatenate all chunks of text into one string to store in MongoDB
                pdf_text = "\n".join([doc.page_content for doc in documents])

                # Save extracted PDF text into MongoDB
                insert_pdf_data(pdf_text)

                st.success("Document uploaded and processed. You can now ask questions about its content.")


    # Question input and response
    question = st.text_input("Ask queries related to the uploaded knowledge:")
    if st.button("Submit query"):
        with st.spinner("Getting your answer..."):
            retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            answer = qa.invoke(question)
            insert_question(question, answer, feedback= "") 
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


            insert_question(st.session_state.st.session_state.answer, answer, st.session_state.feedback_text)
            
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
