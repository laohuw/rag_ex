import os
from llama_cpp import llama, Llama
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from langchain_huggingface import HuggingFaceEmbeddings

model_path= "/home/hd/py_ex/models/llama-2-7b-chat.Q4_K_M.gguf"
CONTEXT_SIZE=512
ngl=20

if __name__=='__main__':

    llama2 = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2000,
        n_ctx=4096,
        verbose=False
    )
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = []
    for file in os.listdir("../docs"):
      if file.endswith(".txt"):
        loader = TextLoader(os.path.join("../docs", file))
        documents.extend(loader.load())
    chunks = text_splitter.split_documents(documents)
    db=FAISS.from_documents(chunks, embedding_model)
    retriever=db.as_retriever()
    # template = """
    # Answer the question based on the following context:
    #
    # {context}
    #
    # Question: {question}
    # Answer:
    # """
    #
    # prompt = PromptTemplate(
    #     template=template,
    #     input_variables=["context", "question"]
    # )

    qa_chain=RetrievalQA.from_chain_type(
        llm=llama2,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": prompt}
    )

    query="What is RAG and how does it work?"
    if(qa_chain is not None):
        result=qa_chain.invoke({"query", query})
        print("Answer", result["result"])
