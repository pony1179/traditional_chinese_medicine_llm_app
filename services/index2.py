from flask import Flask, request, Response # type: ignore
from flask_cors import CORS # type: ignore
from db.weaviate import client
from weaviate.classes.query import MetadataQuery # type: ignore
from llm.init_llm import init_qwen
import torch # type: ignore
from transformers import TextStreamer, TextIteratorStreamer # type: ignore
import threading

collection_name = 'ChineseHerbalMedicinePairs'
collection = client.collections.get(collection_name)

qwen_models = init_qwen()

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持

class CustomTextStreamer:
    def __init__(self, tokenizer, event):
        """
        封装 TextIteratorStreamer，提供流式文本生成
        """
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        self.event = event

    def start_streaming(self, model, inputs):
        """
        在子线程中启动模型生成，并将结果发送到 Streamer
        """
        thread = threading.Thread(target=model.generate, kwargs={
            "input_ids": inputs.input_ids,
            "temperature": 0.7,  # 控制随机性（越低越保守）
            "top_k":1,         # 限制最高概率的50个标记
            "top_p":0.9,       #
            "max_new_tokens": 300,
            "do_sample": True,
            "repetition_penalty":1.4,
            "streamer": self.streamer  # 这里直接传入 streamer，模型会自动填充数据
        })
        thread.start()

    def get_text_stream(self):
        """
        逐步读取生成的文本
        """
        for text in self.streamer:  # `TextIteratorStreamer` 是可迭代的
            yield f"{text}"  # 以 SSE 格式返回
            self.event.set()  # 触发事件，通知有新数据

def generate_answer(tokenizer, model, device, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    event = threading.Event()
    streamer = CustomTextStreamer(tokenizer, event)  # 创建 TextStreamer 对象
    streamer.start_streaming(model, inputs)
    return streamer.get_text_stream()
@app.route('/ask', methods=['POST'])
async def ask_question():
    data = request.json
    query = data["question"]
    model_name = data["model"]
    use_external_db = data.get("external_db", True)  # 默认为 True
    print("当前使用的模型:",model_name)
    [tokenizer, model, device] = qwen_models[model_name]
    context = ''
    input_text = f"Question: {query}\n 请给我中文回答。Answer:"
    if use_external_db:
        # 使用知识库逻辑
        results = collection.query.near_text(
            query=query,
            limit=10,
            return_metadata=MetadataQuery(distance=True)
        )
        for o in results.objects:
            properties = o.properties
            context = context + f"'{properties['type']}  {properties['medicine_a']}  {properties['medicine_b']}  {properties['dosage']}  {properties['function']}' \n"
    if (len(context)): 
        input_text = f"Context: {context}\nQuestion: {query}\n 请给我中文回答。Answer:"
    return Response(generate_answer(tokenizer, model, device, input_text), content_type="text/plain")

if __name__ == "__main__":
    app.run(port=5001)