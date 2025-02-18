from os.path import exists

import langchain
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def data_extraction(index, urls, embeddings, main_placeholder=None):
    """
    Extracts data from the given URLs, splits the data into chunks, and saves the chunks into a FAISS index.

    Args:
        index (str): The name of the FAISS index to save.
        urls (list): List of URLs to load data from.
        embeddings (OllamaEmbeddings): The embeddings model to use.
        main_placeholder (streamlit.DeltaGenerator, optional): Streamlit placeholder for displaying status messages.
    """
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    print("text splitter", text_splitter)
    if main_placeholder:
        main_placeholder.text("Splitting data into chunks...")
    else:
        print("Splitting data into chunks...")
    docs = text_splitter.split_documents(data)
    vector_store = FAISS.from_documents(docs, embeddings)
    if main_placeholder:
        main_placeholder.text("Saving index...")
    else:
        print("Saving index...")
    vector_store.save_local(index)


def get_response(index, llm, embeddings, query):
    """
    Retrieves a response to the given query using the FAISS index and the LLM.

    Args:
        index (str): The name of the FAISS index to load.
        llm (OllamaLLM): The language model to use.
        embeddings (OllamaEmbeddings): The embeddings model to use.
        query (str): The query to retrieve a response for.

    Returns:
        dict: The response from the language model.
    """
    if exists(index):
        vector_store = FAISS.load_local(index, embeddings, allow_dangerous_deserialization=True)

        custom_prompt_template = """Use the following retrieved documents to answer the question.
        If the answer is not contained in the documents, say "I don't know."
        Provide sources at the end.

        Question: {question}

        Documents:
        {context}
        
        Summaries:
        {summaries}

        Answer:
        """

        # Create a custom prompt
        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["question", "context", "summaries"],
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": custom_prompt},
        )
        langchain.debug = True
        result = chain.invoke({"question": query, "context": "", "summaries": ""}, return_only_outputs=True)
        return result


if __name__ == '__main__':
    """
    Main function to run the Streamlit app for question answering with sources.
    """
    st.title("Question Answering with Sources")
    st.sidebar.title("News article URL")
    index_name = "faiss_index"
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}")
        urls.append(url)
    submit_pressed = st.sidebar.button("Submit")
    main_placeholder = st.empty()

    llm = OllamaLLM(
        temperature=0.6,
        model='llama3.2:latest'
    )

    embeddings = OllamaEmbeddings(
        model='nomic-embed-text:latest'
    )

    if submit_pressed:
        main_placeholder.text("Loading data from URLs...")
        data_extraction(index_name, urls, embeddings, main_placeholder)

    query = main_placeholder.text_input("Question?: ")
    if query:
        res = get_response(index_name, llm, embeddings, query)
        st.header("Answer:")
        st.subheader(res['answer'])

#  Debugging code
# if __name__ == '__main__':
#     llm = OllamaLLM(
#         temperature=0.6,
#         model='llama3.2:latest'
#     )

#     embeddings = OllamaEmbeddings(
#         model='nomic-embed-text:latest'
#     )
#     index_name = "faiss_index"
#     urls = ["https://wiki.python.org/moin/BeginnersGuide/Download"]

#     data_extraction(index_name, urls, embeddings)
#     res = get_response(index_name, llm, embeddings, "How to install python in ubuntu?")
#     print(res)
