import pandas as pd

df = pd.read_csv("../data/data_sets/桂林古本伤寒杂病论药方.train_副本.csv")
inputs = df["input"].tolist()
outputs = df["output"].tolist()
# 保存到新的 csv 文件
list_to_insert = []
for i in range(len(inputs)):
    if len(inputs[i]) > 0 and len(outputs[i]) > 0:
        # 方剂组成：以后，制备方法：以前的字符串进行切割
        if "方剂组成：" in inputs[i] and "制备方法：" in inputs[i]:
            list_to_insert.append({"input": inputs[i].split("方剂组成：")[1].split("制备方法：")[0].replace("\n", ""), "output": outputs[i].split("方剂名称：")[1]})
pd.DataFrame(list_to_insert, columns=["input", "output"]).to_csv("../data/data_sets/桂林古本伤寒杂病论药方.train_new.csv", index=False)