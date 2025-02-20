import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
def connect_to_qwen(model_name):
    print("正在连接模型：", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model = model.to(device)
    return tokenizer, model, device
