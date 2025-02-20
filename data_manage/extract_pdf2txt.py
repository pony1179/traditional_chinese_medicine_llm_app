import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置嵌入生成
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
def process_pdfs(directory):
    # 遍历指定目录及其子目录中的所有文件
    for root, _, files in os.walk(directory):
        # 遍历当前目录中的所有文件
        for file in files:
            # 检查文件是否以".pdf"结尾
            if file.endswith(".pdf"):
                # 创建一个PDF加载器实例，用于加载PDF文件
                loader = PyPDFLoader(os.path.join(root, file))
                # 使用文本分割器加载并分割PDF文档
                documents = loader.load_and_split(text_splitter)
                # 判断文件是否存在
                if not os.path.exists('../data/extracted_books/' + file.split('.')[0] + '.txt'):
                    with open('../data/extracted_books/' + file.split('.')[0] + '.txt', 'w', encoding='utf8') as file:
                        for doc in documents:
                            file.write(doc.page_content)
try:
    process_pdfs("../data/books")
finally:
    print("done")