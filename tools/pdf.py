from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from models.models import ModelFactory
import os


class PdfTools:
    _vectorstore = Chroma(embedding_function=HuggingFaceEmbeddings(), persist_directory=os.getenv("CHROMA_DB_PATH"))

    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    _qa_chain = RetrievalQA.from_chain_type(
        ModelFactory.public_model,
        retriever=_vectorstore.as_retriever()
    )

    @tool
    def pdf_qa(questions: list[str], pdf_files: list[str] = []):
        """reads multiple pdf files and answers the questions. If the pdf files is empty, then answers questions based on the previously stored pdf data"""

        print(f"pdf_files: {pdf_files} | questions: {questions}")

        all_splits = []

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print("Error: pdf file {} does not exist. Skipping the file.".format(pdf_file))
                continue

            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            splits = PdfTools._text_splitter.split_documents(pages)
            all_splits.extend(splits)

        if len(all_splits) != 0:
            PdfTools._vectorstore.add_documents(all_splits)

        responses = []

        for query in questions:
            result = PdfTools._qa_chain.invoke(query)

            print(result)

            if result is not None:
                responses.append(result)

        print("Done pdf_qa")

        return f'Here are my responses: {responses}'

    tools = [pdf_qa]