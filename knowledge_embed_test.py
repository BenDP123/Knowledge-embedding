import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence  # Correct import for RunnableSequence

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="DP_Marketo_test_documentation.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

print(len(documents))

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=1)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup RunnableSequence & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4")

template = """
You are a world class Marketo service agent. 
I will share a client request with you and you will give me the best answer that 
how I should fulfill this request based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices, 
in terms of length, logical arguments, and other details.

2/ If the best practice is irrelevant, then try to mimic the style of the best practice to the prospect's message.

Below is a request I received from the client:
{message}

Here is a list of best practices of how we normally respond to prospects in similar scenarios:
{best_practice}

Please write the best way to fulfill the client request:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

# Correct chaining using the pipe operator
runnable_sequence = prompt | llm

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = runnable_sequence.invoke({"message": message, "best_practice": best_practice})
    return response

# 5. Build an app with Streamlit
def main():
    st.set_page_config(page_title="Marketo best practice generator", page_icon=":bird:")
    st.header("Marketo best practice generator :bird:")
    message = st.text_area("Customer message")

    if message:
        st.write("Generating best practice message...")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()
