import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

chroma_client = chromadb.PersistentClient(path="/Users/pony/storage/vector_stores/chromadb")
collection = chroma_client.get_or_create_collection(name="traditional_chinese_medical_knowledge")

# 配置嵌入生成
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# 递归查找 PDF 并存入 Weaviate
def process_pdfs(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load_and_split(text_splitter)
                for doc in documents:
                    count = int(collection.count())
                    collection.upsert(
                        documents=[
                            doc.page_content
                        ],
                        ids=[str(count+1)],
                        metadatas=[
                            doc.metadata
                        ]
                    )

try:
    process_pdfs("/Users/pony/storage/vector_stores/origin_data")
finally:
    # 确保在程序完成后关闭 Weaviate 客户端
    print("done")