import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

# Initialize Pinecone and LangChain components
pinecone.init(api_key=st.secrets['PINECONE_API_KEY'], environment='us-east-1-aws')
index = pinecone.Index('patrundev1')
embed_model = OpenAIEmbeddings(model='text-embedding-ada-002')
vectorstore = Pinecone(index, embed_model.embed_query, "text")

# Initialize ChatOpenAI
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
chat = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model='gpt-3.5-turbo'
)

def upsert_file_to_pinecone(uploaded_file):
    # Read the file and process it
    text = uploaded_file.getvalue().decode("utf-8")
    # Generate embeddings
    embeddings = embed_model.embed_documents([text])
    # Upsert to Pinecone
    vector_id = "unique_vector_id"  # Generate a unique ID for the vector
    index.upsert(vectors=[(vector_id, embeddings[0], {"text": text})])

def chat_with_gpt(prompt):
    messages = [SystemMessage(content='You are a helpful assistant.'),
                HumanMessage(content=prompt)]
    response = chat(messages)
    return response.content

# Streamlit UI
st.title('Pinecone & GPT Chat Application')

# File uploader
uploaded_file = st.file_uploader("Choose a file to upload to Pinecone", type=['txt'])
if uploaded_file is not None:
    upsert_file_to_pinecone(uploaded_file)
    st.success('File successfully uploaded and upserted to Pinecone!')

# Chat interface
user_input = st.text_input("Talk to GPT:")
if user_input:
    gpt_response = chat_with_gpt(user_input)
    st.text_area("GPT Response:", value=gpt_response, height=300, disabled=True)