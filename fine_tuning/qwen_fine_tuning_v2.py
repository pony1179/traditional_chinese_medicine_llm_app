from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# 设备检测
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据集
dataset = load_dataset('csv', data_files={'train': '../data/data_sets/桂林古本伤寒杂病论药方.train.csv', 'validation': '../data/data_sets/桂林古本伤寒杂病论药方.val.csv'})

# 初始化模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
model = model.to('mps')
# model = torch.compile(model)
model.config.use_cache = False  # 禁用 use_cache
model.gradient_checkpointing_enable()
# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 适用于因果语言建模（CausalLM）
    r=32,  # 低秩适配参数
    lora_alpha=32,  # 缩放系数
    lora_dropout=0.05,  # Dropout 以防止过拟合
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "gate_proj", "down_proj", "up_proj"])

# 应用 LoRA 适配
model = get_peft_model(model, lora_config)

# 数据预处理
def preprocess_function(examples):
    """预处理数据，确保输入格式符合任务需求"""
    inputs = examples["input"]
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
    )["input_ids"]

    # 忽略 padding token 计算损失，避免影响梯度
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

# 对数据集进行 tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,  # 使用较小的学习率，避免遗忘预训练知识
    per_device_train_batch_size=8,  # batch size 适中
    per_device_eval_batch_size=8,
    num_train_epochs=20,  # 适中 epoch 避免过拟合
    weight_decay=0.005,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=50,  # 频繁记录日志，检查损失变化
    save_total_limit=2,  # 仅保留最近的 2 个模型检查点
    fp16=False,  # MPS 不支持 fp16
    bf16=True,  # MPS 适用 bfloat16 进行计算
    gradient_checkpointing=True,  # ✅ 启用梯度检查点，减少显存占用
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8  # 让 Tensor 变成 8 的倍数，提高计算效率
)

# 训练
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

# 保存模型
trainer.save_model("Qwen/tcm_finetuned_qwen")
tokenizer.save_pretrained("Qwen/tcm_finetuned_qwen")

print("训练完成，微调后的模型已保存！")