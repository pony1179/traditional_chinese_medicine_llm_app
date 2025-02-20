import weaviate
# 初始化 Weaviate 客户端
def init_weaviate_client():
    # 创建 Weaviate 客户端
    client = weaviate.connect_to_local(
        host="127.0.0.1",  # Use a string to specify the host
        port=8080,
        grpc_port=50051,
    )
    if client.is_ready():
        print("Weaviate connected successfully")
    else:
        print("Weaviate connection failed")
    return client

# 全局 Weaviate 客户端对象
client = init_weaviate_client()