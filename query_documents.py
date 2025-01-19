import json
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest_documents import deserialize_document, serialize_document


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

def format_docs(docs: List[Document | bytes]) -> str:
    """Format documents into a single string"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        if isinstance(doc, bytes):
            doc = deserialize_document(doc.decode())
        formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

def create_chain():
    """Create a chain that combines retrieval and generation"""
    retriever = setup_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following documents:
    
    {context}
    
    Question: {question}
    
    Answer: Let me help you with that.
    """)
    
    # Create the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def query_documents(query: str):
    """Query the documents and generate an answer"""
    chain = create_chain()
    
    print(f"\nQuestion: {query}")
    print("\nThinking...")
    
    # Get the answer
    answer = chain.invoke(query)
    
    print("\nAnswer:")
    print(answer)
    
    # Also show the retrieved documents
    print("\nRelevant Documents:")
    retrieved_docs = setup_retriever().get_relevant_documents(query)
    for i, doc_bytes in enumerate(retrieved_docs, 1):
        if isinstance(doc_bytes, bytes):
            doc = deserialize_document(doc_bytes.decode())
        else:
            doc = doc_bytes
        print(f"\nDocument {i}:")
        print(doc.page_content[:200] + "...")
    
    return answer, retrieved_docs

def main():
    """Main function with command line argument handling"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python query_documents.py 'your question here'")
        return
    
    query = " ".join(sys.argv[1:])
    query_documents(query)

if __name__ == "__main__":
    main() 