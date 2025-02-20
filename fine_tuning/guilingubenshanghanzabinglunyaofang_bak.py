from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import sys
sys.path.append('../')
import torch
torch.mps.empty_cache()
torch.set_float32_matmul_precision('high')  # 或者 'high'

# 加载数据
dataset = load_dataset('csv', data_files={'train': '../data/data_sets/桂林古本伤寒杂病论药方.train.csv', 'validation': '../data/data_sets/桂林古本伤寒杂病论药方.val.csv'})

# 初始化模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# tokenizer, model, device = connect_to_qwen(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
model = model.to('mps')
# model = torch.compile(model)
model.config.use_cache = False  # 禁用 use_cache
model.gradient_checkpointing_enable()
tokenizer.save_pretrained("Qwen/tcm_finetuned_qwen")
# 数据预处理
def preprocess_function(examples):
    inputs = [f"方剂组成：{ex}\n请给出该方剂的名称。" for ex in examples["input"]]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    # fp16=True,
    predict_with_generate=True,
    gradient_checkpointing=True,  # ✅ 启用梯度检查点，减少显存占用
    fp16=False,  # MPS 不支持 fp16，但如果有 NVIDIA GPU 可启用
    bf16=True,  # ✅ MPS 上可以使用 `bfloat16` 进行计算加速
    logging_dir='./logs',  # 记录日志
    logging_steps=100,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8  # ✅ 让 Tensor 变成 8 的倍数，提升计算效率
)

# 创建Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("Qwen/tcm_finetuned_qwen")
tokenizer.save_pretrained("Qwen/tcm_finetuned_qwen")