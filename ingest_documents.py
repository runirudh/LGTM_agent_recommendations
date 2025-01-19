import uuid
import json
from typing import List
from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore

def deserialize_document(doc_str: str) -> Document:
    """Deserialize a JSON string back to Document object"""
    data = json.loads(doc_str)
    return Document(
        page_content=data["page_content"],
        metadata=data["metadata"]
    )

def serialize_document(doc: Document) -> str:
    """Serialize a Document object to JSON string"""
    return json.dumps({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    })

class HypotheticalQuestions(BaseModel):
    """Schema for hypothetical questions output"""
    questions: List[str] = Field(description="List of hypothetical questions")

def load_and_split_documents():
    """Load and split documents"""
    # Load initial documents
    loader = WebBaseLoader([
        "https://raw.githubusercontent.com/hwchase17/chroma-langchain/refs/heads/master/state_of_the_union.txt",
        "https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt"
    ])
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    
    # Split while preserving metadata
    sub_docs = []
    for doc, doc_id in zip(docs, doc_ids):
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
        sub_docs.extend(chunks)
    
    return docs, sub_docs, doc_ids

def generate_summaries(docs):
    """Generate summaries for documents"""
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
    return chain.batch(docs)

def generate_questions(docs):
    """Generate hypothetical questions"""
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Generate 3 hypothetical questions this document could answer:\n\n{doc}"
        )
        | ChatOpenAI(model="gpt-4o-mini").with_structured_output(HypotheticalQuestions)
        | (lambda x: x.questions)
    )
    return chain.batch(docs)

def setup_retriever():
    """Initialize and return the retriever with existing stores"""
    docstore = RedisStore(
        redis_url="redis://localhost:6379?socket_timeout=5&socket_connect_timeout=5&retry_on_timeout=true",
        namespace="docstore",
    )
    vectorstore = Chroma(
        collection_name="multi_vector_store",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db"
    )
    
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        child_key="child_id",
        serializer=serialize_document,
        deserializer=deserialize_document
    )

def main():
    """Handle document ingestion process"""
    retriever = setup_retriever()
    
    # Load and process documents
    print("Loading and splitting documents...")
    docs, sub_docs, doc_ids = load_and_split_documents()
    
    # Store original documents (serialized)
    print("Storing original documents...")
    doc_pairs = [(doc_id, serialize_document(doc)) 
                 for doc_id, doc in zip(doc_ids, docs)]
    retriever.docstore.mset(doc_pairs)
    
    # Add split documents to vectorstore
    print("Adding split documents...")
    retriever.vectorstore.add_documents(sub_docs)
    
    # Generate and add summaries
    print("Generating summaries...")
    summaries = generate_summaries(docs)
    summary_docs = [
        Document(page_content=summary, metadata={"doc_id": doc_id})
        for summary, doc_id in zip(summaries, doc_ids)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    
    # Generate and add questions
    print("Generating questions...")
    all_questions = generate_questions(docs)
    question_docs = []
    for questions, doc_id in zip(all_questions, doc_ids):
        question_docs.extend([
            Document(page_content=question, metadata={"doc_id": doc_id})
            for question in questions
        ])
    retriever.vectorstore.add_documents(question_docs)
    print("Ingestion complete!")

if __name__ == "__main__":
    main() 