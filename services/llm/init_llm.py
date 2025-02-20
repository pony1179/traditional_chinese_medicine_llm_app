from .qwen import connect_to_qwen

qwen_model_names = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "/Users/pony/work/own/llm/langchain/traditional_chinese_medicine_llm_app/fine_tuning/Qwen/tcm_finetuned_qwen"
]

def init_qwen():
    qwen_models = {}
    """
    初始化指定版本的通义千问模型。
    """
    for model_name in qwen_model_names:
        tokenizer, model, device = connect_to_qwen(model_name)
        qwen_models[model_name] = [tokenizer, model, device]
    # 加载模型和分词器
    
    return qwen_models
