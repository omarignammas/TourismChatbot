import os
import re
from itertools import chain
import pandas as pd
from io import StringIO
from langchain_community.document_loaders import UnstructuredMarkdownLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import Document, HumanMessage, SystemMessage


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def load_markdown_text(root_path):
    processed_files = []
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if file.startswith("output") and file.endswith(".md"):
                file_path = os.path.join(root,file)
                loader = UnstructuredMarkdownLoader(
                    file_path,
                    mode="single",
                    strategy="fast",
                )
                doc = loader.load()
                cleaned_doc = clean_lines(doc[0].page_content)
                processed_files.append(cleaned_doc)
    return processed_files

def clean_lines(long_text):
    long_text = re.sub(r'\n\n[a-z]+', '', long_text)
    long_text = re.sub(r'\t', '', long_text)
    long_text = re.sub(r'\s{2,}', ' ', long_text)
    long_text = re.sub(r'\n{3,}', '\n\n', long_text)
    return long_text.strip()

def load_csv(root_path):
    all_tables = []
    for dir_path,dirs,files in os.walk(root_path):
        for file in files:
            if file.startswith("evolution") and file.endswith(".md"):
                file_path = os.path.join(dir_path,file)
                with open(file_path,"r",encoding="utf-8") as f:
                    md_table = f.read()
                df_table = convert_to_csv(md_table)
                csv_path = os.path.join(dir_path,file.split('.')[0]+'.csv')
                df_table.to_csv(csv_path,index=False)

                loader_csv = CSVLoader(csv_path)
                table_elements = loader_csv.load()
                all_tables.append(table_elements)
    return all_tables

def convert_to_csv(md_table):
        md_table_clean = md_table.replace("<br>", " ")
        df = pd.read_csv(StringIO(md_table_clean), sep="|", engine="python").iloc[:,1:-1]
        return df

def chunk_texts(text_list):
    all_chunks = []
    for text in text_list:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700,  separators=["\n\n", "\n", " "], chunk_overlap=20)
        chunks = splitter.split_text(text)
        all_chunks.append(chunks)
    return all_chunks

def store_vectors(embedding,chunks):
    vector_store = FAISS.from_texts(chunks, embedding)
    '''qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",#determines how the retrieved documents are sent to the LLM.
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa_chain'''
    return vector_store

def load_text(file_path):
    text_loader = TextLoader(file_path,encoding="utf-8")
    text_content = text_loader.load()
    return text_content

def chunk_doc(doc_content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, separators=["\n\n", "\n", " "], chunk_overlap=10)
    chunks = splitter.split_text(doc_content[0].page_content)
    return chunks

if __name__ == "__main__":
    root_path = r"C:\Users\HUAWEI\Desktop\Project_me_and_zineb\data\Regions\Tanger_Hoceima"
    files_content = load_markdown_text(root_path) #will be chunked
    tables_content = load_csv(root_path) #not going to be chunked again

    file_path = r".\data_processed.txt"
    chunks1 = chunk_texts(files_content)
    file_content = load_text(file_path)
    chunks2 = chunk_doc(file_content)

    all_chunks = list(chain.from_iterable(chunks1)) + chunks2
    vector_store = store_vectors(embeddings,all_chunks)
    vector_store.save_local("faiss_index3")

