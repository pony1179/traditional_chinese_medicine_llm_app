"""
基于 Qwen 模型的中医古籍微调训练脚本
此脚本用于对 Qwen2.5-0.5B-Instruct 模型进行微调，以适应中医古籍文本处理任务

主要功能：
1. 加载和预处理数据集
2. 配置模型和 tokenizer
3. 设置 LoRA 参数进行高效微调
4. 训练模型并保存结果
"""

import logging
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class QwenFineTuner:
    """Qwen 模型微调类"""

    def __init__(
        self,
        model_name="Qwen2.5-0.5B-Instruct",
        use_mirror=False,
        checkpoint_dir="./results",
        final_model_dir="./models/tcm_finetuned_qwen",
    ):
        """
        初始化微调器

        参数:
            model_name (str): 预训练模型的名称
            use_mirror (bool): 是否使用镜像站点
            checkpoint_dir (str): 训练过程中的检查点保存路径
            final_model_dir (str): 最终模型保存路径
        """
        # 检查是否包含组织名，如果没有，添加默认组织名
        if "/" not in model_name:
            self.model_name = f"Qwen/{model_name}"
        else:
            self.model_name = model_name

        self.use_mirror = use_mirror
        self.checkpoint_dir = checkpoint_dir
        self.final_model_dir = final_model_dir
        self.device = self._detect_device()
        self.tokenizer = None
        self.model = None

        # 设置环境变量，决定是否使用镜像
        if not use_mirror:
            # 如果环境变量已经设置，则暂时保存它
            self.original_hf_endpoint = os.environ.get("HF_ENDPOINT", "")
            # 确保使用官方 API
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"

        # 创建保存目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)

        logger.info(f"初始化 QwenFineTuner，使用设备: {self.device}")
        logger.info(f"模型名称: {self.model_name}")
        logger.info(f"使用镜像: {'是' if use_mirror else '否'}")
        logger.info(f"检查点保存路径: {self.checkpoint_dir}")
        logger.info(f"最终模型保存路径: {self.final_model_dir}")

    def __del__(self):
        """
        析构函数，恢复环境变量
        """
        if hasattr(self, "original_hf_endpoint") and not self.use_mirror:
            if self.original_hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.original_hf_endpoint
            else:
                # 如果原来没有设置，则删除它
                os.environ.pop("HF_ENDPOINT", None)

    def _detect_device(self):
        """
        检测并返回可用的计算设备

        返回:
            str: 'mps', 'cuda' 或 'cpu'
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _check_model_exists(self):
        """
        检查模型是否存在于 Hugging Face Hub

        返回:
            bool: 模型是否存在
        """
        try:
            logger.info(f"正在检查模型 {self.model_name} 是否存在...")
            info = model_info(self.model_name)
            logger.info(f"模型信息：")
            logger.info(f"- 模型ID: {info.modelId}")
            logger.info(f"- 最后更新: {info.lastModified}")
            logger.info(f"- 标签: {', '.join(info.tags) if info.tags else '无'}")
            logger.info(f"- 下载量: {info.downloads:,}")
            logger.info(f"- 点赞数: {info.likes:,}")
            return True
        except RepositoryNotFoundError:
            logger.error(f"错误：模型 {self.model_name} 在 Hugging Face Hub 上不存在")
            logger.error(f"请检查模型名称是否正确。常见的 Qwen 模型名称包括:")
            logger.error(f"- Qwen/Qwen2.5-0.5B-Chat")
            logger.error(f"- Qwen/Qwen2.5-1.8B-Chat")
            logger.error(f"- Qwen/Qwen2.5-4B-Chat")
            logger.error(f"- Qwen/Qwen2.5-7B-Chat")
            logger.error(f"- Qwen/Qwen2.5-14B-Chat")
            logger.error(f"- Qwen/Qwen2.5-32B-Chat")
            logger.error(f"- Qwen/Qwen2.5-72B-Chat")
            return False
        except HfHubHTTPError as e:
            logger.error(f"错误：无法访问 Hugging Face Hub：{str(e)}")
            logger.error(f"尝试使用官方 API: 设置 use_mirror=False")
            return False
        except Exception as e:
            logger.error(f"错误：检查模型时发生未知错误：{str(e)}")
            return False

    def load_dataset(self, train_path, val_path):
        """
        加载训练和验证数据集

        参数:
            train_path (str): 训练数据集路径
            val_path (str): 验证数据集路径

        返回:
            Dataset: 包含训练和验证集的数据集对象
        """
        logger.info("开始加载数据集...")
        logger.info(f"训练集路径: {train_path}")
        logger.info(f"验证集路径: {val_path}")

        # 检查文件是否存在
        if not os.path.exists(train_path):
            logger.error(f"错误：训练集文件不存在: {train_path}")
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")

        if not os.path.exists(val_path):
            logger.error(f"错误：验证集文件不存在: {val_path}")
            raise FileNotFoundError(f"验证集文件不存在: {val_path}")

        try:
            dataset = load_dataset(
                "csv", data_files={"train": train_path, "validation": val_path}
            )

            logger.info(f"数据集加载完成！")
            logger.info(f"训练集大小: {len(dataset['train'])} 条")
            logger.info(f"验证集大小: {len(dataset['validation'])} 条")

            # 检查数据集结构
            logger.info(f"数据集列名: {dataset['train'].column_names}")

            # 验证必要的列是否存在
            if "formatted" not in dataset["train"].column_names:
                logger.error(f"错误：数据集缺少必要的列 'formatted'")
                raise ValueError(f"数据集格式错误：缺少必要的列 'formatted'")

            return dataset
        except Exception as e:
            logger.error(f"加载数据集时发生错误: {str(e)}")
            raise

    def setup_model(self):
        """
        设置并配置模型和 tokenizer
        """
        logger.info(f"开始加载模型和tokenizer: {self.model_name}")

        # 检查模型是否存在
        if not self._check_model_exists():
            # 尝试替代模型
            alt_model = "Qwen/Qwen2.5-0.5B-Chat"
            logger.warning(f"原始模型不存在，尝试使用替代模型: {alt_model}")
            self.model_name = alt_model
            if not self._check_model_exists():
                raise ValueError(f"模型 {self.model_name} 不存在或无法访问")

        # 初始化 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            logger.info("Tokenizer 加载完成")
        except Exception as e:
            logger.error(f"加载 tokenizer 时发生错误：{str(e)}")
            raise

        # 加载预训练模型
        try:
            logger.info("开始加载预训练模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            logger.info("预训练模型加载完成")
        except Exception as e:
            logger.error(f"加载预训练模型时发生错误：{str(e)}")
            raise

        # 配置模型训练参数
        self.model.config.use_cache = False  # 禁用 KV cache 以支持梯度检查点
        self.model.gradient_checkpointing_enable()
        logger.info("模型训练参数配置完成")

        # 配置 LoRA
        logger.info("开始配置 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # LoRA 秩
            lora_alpha=32,  # LoRA alpha 参数
            lora_dropout=0.05,  # LoRA dropout 率
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "fc1",
                "fc2",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        )

        # 应用 LoRA 配置
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA 配置完成")

        # 打印可训练参数信息
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数数量: {trainable_params:,}")
        logger.info(f"总参数数量: {all_params:,}")
        logger.info(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")

    def preprocess_function(self, examples):
        """
        预处理数据集示例

        参数:
            examples (dict): 包含原始文本的示例

        返回:
            dict: 处理后的模型输入
        """
        inputs = examples["formatted"]

        # Tokenize 输入文本
        model_inputs = self.tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )

        # 处理标签
        labels = self.tokenizer(
            examples["formatted"],
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )["input_ids"]

        # 将填充 token 的标签设为 -100（在损失计算时忽略）
        labels = [
            [
                (token if token != self.tokenizer.pad_token_id else -100)
                for token in label
            ]
            for label in labels
        ]

        model_inputs["labels"] = labels
        return model_inputs

    def get_training_args(self):
        """
        配置训练参数

        返回:
            Seq2SeqTrainingArguments: 训练参数对象
        """
        logger.info("配置训练参数...")
        args = Seq2SeqTrainingArguments(
            output_dir=self.checkpoint_dir,  # 使用配置的检查点路径
            eval_strategy="epoch",  # 评估策略
            save_strategy="epoch",  # 保存策略
            learning_rate=5e-6,  # 学习率
            per_device_train_batch_size=8,  # 训练批次大小
            per_device_eval_batch_size=8,  # 评估批次大小
            num_train_epochs=1,  # 训练轮数
            weight_decay=0.005,  # 权重衰减
            predict_with_generate=True,  # 预测时使用生成
            logging_dir="./logs",  # 日志保存路径
            logging_steps=50,  # 日志保存步数
            save_total_limit=2,  # 最大保存模型数量
            fp16=False,  # MPS 不支持 fp16
            bf16=True,  # 使用 bfloat16 精度
            gradient_checkpointing=True,
            report_to="none",
        )
        logger.info(f"训练参数配置完成: {args}")
        return args

    def train(self, dataset):
        """
        训练模型

        参数:
            dataset (Dataset): 预处理后的数据集
        """
        logger.info("开始训练准备...")

        # 数据整理器配置
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, pad_to_multiple_of=8
        )
        logger.info("数据整理器配置完成")

        # 配置训练器
        logger.info("配置训练器...")
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        logger.info("训练完成！")

        # 保存模型和 tokenizer
        output_dir = self.final_model_dir
        logger.info(f"保存模型到: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("模型保存完成！")


def main():
    """主函数"""
    logger.info("=== 开始中医古籍 Qwen 模型微调训练 ===")

    # 初始化微调器，设置模型保存路径
    fine_tuner = QwenFineTuner(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        # model_name="Qwen/Qwen2.5-0.5B-Instruct",
        use_mirror=True,
        checkpoint_dir="./checkpoints/qwen_tcm",  # 检查点保存路径
        final_model_dir="./models/tcm_qwen_v1",  # 最终模型保存路径
    )

    # 加载数据集
    dataset = fine_tuner.load_dataset(
        "../data/data_sets/桂林古本伤寒杂病论药方.train_v4.csv",
        "../data/data_sets/桂林古本伤寒杂病论药方.val_v4.csv",
    )

    # 设置模型
    fine_tuner.setup_model()

    # 预处理数据集
    logger.info("开始数据预处理...")
    tokenized_datasets = dataset.map(
        fine_tuner.preprocess_function, batched=True, desc="预处理数据集"
    )
    logger.info("数据预处理完成")

    # 训练模型
    fine_tuner.train(tokenized_datasets)

    logger.info("=== 训练流程全部完成！===")


if __name__ == "__main__":
    main()
