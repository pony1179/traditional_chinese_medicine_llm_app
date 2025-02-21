import random
import pandas as pd

def augment_input(base_text):
    variations = [
        f"请根据以下药材组成预测方剂名称：{base_text}",
        f"这个方剂由 {base_text} 组成，请推测其名称。",
        f"以下药方包含：{base_text}，请问它叫什么？",
        f"该方剂主要成分为：{base_text}，你能猜出它的名字吗？",
        f"请告诉我，{base_text} 对应的方剂名称是什么？"
    ]
    return random.choice(variations)
def augment_output(output):
    variations = [
        f"该方剂名称是：{output}",
        f"这个方剂被称为 {output}",
        f"它的名字是 {output}",
        f"{output} 是这个方剂的名称",
        f"正确答案是 {output}",
        output  # 原始输出
    ]
    return random.choice(variations)

df = pd.read_csv("../data/data_sets/桂林古本伤寒杂病论药方.train_v3.csv")
inputs = df["input"].tolist()
outputs = df["output"].tolist()
# 保存到新的 csv 文件
for i in range(len(inputs)):
    list_to_insert = []
    for index in range(5):
        toInsertInput = augment_input(inputs[i])
        toInsertOutput = augment_output(outputs[i])
        list_to_insert.append({"input":toInsertInput, "output": toInsertOutput, "formatted": f"<|im_start|>system\n你是一个中医助手，帮助用户匹配方剂名称。\n<|im_end|>\n"
                f"<|im_start|>user\n{toInsertInput}<|im_end|>\n"
                f"<|im_start|>assistant\n{toInsertOutput}<|im_end|>"})
    pd.DataFrame(list_to_insert, columns=["input", "output",  "formatted"]).to_csv("../data/data_sets/桂林古本伤寒杂病论药方.train_v4.csv", mode="a",header=False, index=False)