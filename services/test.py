from flask import Flask, request, Response
from flask_cors import CORS
from db.weaviate import client
from weaviate.classes.query import MetadataQuery # type: ignore
from llm.init_llm import init_qwen
import torch # type: ignore
from transformers import TextStreamer # type: ignore
import threading

collection_name = 'ChineseHerbalMedicinePairs'
collection = client.collections.get(collection_name)

qwen_models = init_qwen()

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持
class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, event, skip_prompt=False):
        super().__init__(tokenizer, skip_prompt)
        self.generated_text = ""  # 用于存储生成的文本
        self.event = event
        self.new_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        重写父类方法，每次生成新的文本时调用。
        """
        self.generated_text += text
        self.new_text = text
        print(text)
        self.event.set()

def generate_answer(tokenizer, model, device, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # `streamer` 让模型逐步输出
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    with torch.no_grad():
        model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,  # 控制随机性（越低越保守）
            streamer=streamer
        )
        while True:
            chunk = streamer.text_queue.get()  # 取出一段新生成的文本
            if chunk is None:  # 结束标志
                break
            yield f"data: {chunk}\n\n"  # SSE 格式返回


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
    return Response(generate_answer(tokenizer, model, device, input_text), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(port=5001)