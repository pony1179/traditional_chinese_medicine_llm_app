import os
import weaviate
import weaviate.classes as wvc
from datetime import datetime

# 配置嵌入生成

# 初始化 Weaviate 客户端
client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)
print(client.is_ready())
types = [
    '相须药对', '相反药对-扶正祛邪相反药对', '相反药对-寒热相反药对', 
    '相反药对-升降相反药对', '相反药对-润燥相反药对', '相反药对-散敛相反药对',
    '相反药对-其它相反药对', '相使药对', '十八反、十九畏药对'
]
collection_name = 'ChineseHerbalMedicinePairs'
# 创建一个中药药对集合
if not client.collections.exists(collection_name):
    ChineseHerbalMedicinePairsCollection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="medicine_a",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="medicine_b",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="dosage",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="function",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="type",
                data_type=wvc.config.DataType.TEXT,
            )
        ]
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")

ChineseHerbalMedicinePairsCollection = client.collections.get(collection_name)

# 递归查找 PDF 并存入 Weaviate
def extract_data(filePath):
    data_type = ''
    with open(filePath, 'r', encoding='utf8') as file:
        while True:
            line = file.readline().strip()
            if line in types:
                data_type = line
                continue
            if not line:  # 如果 readline() 返回空字符串，说明已到达文件末尾
                break
            arr = line.split('  ')
            response = ChineseHerbalMedicinePairsCollection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("medicine_a").equal(arr[0])
                & wvc.query.Filter.by_property("medicine_b").equal(arr[1])
            )
            if response.objects:
                uuid = response.objects[0].uuid  # 获取对象的 ID
                ChineseHerbalMedicinePairsCollection.data.update(
                    uuid=uuid,
                    properties={
                        "medicine_a": arr[0],
                        "medicine_b": arr[1],
                        "dosage": arr[2],
                        "function": arr[3],
                        "type": data_type
                    }
                )
                print(f"'{uuid}'数据已更新。")
            else:
                print(arr)
                uuid = ChineseHerbalMedicinePairsCollection.data.insert(
                    properties={
                        "medicine_a": arr[0],
                        "medicine_b": arr[1],
                        "dosage": arr[2],
                        "function": arr[3],
                        "type": data_type
                    }
                )
                print(f"Inserted object with UUID: {uuid}")

            
try:
    extract_data("./books/中药药对.txt")
finally:
    # 确保在程序完成后关闭 Weaviate 客户端
    print("done")
    client.close()