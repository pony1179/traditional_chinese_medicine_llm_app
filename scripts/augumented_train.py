import os
import re
import pandas as pd
from openai import OpenAI

# 读取 API Key（从环境变量）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_more_data(df):
    inputs = df["input"].tolist()
    outputs = df["output"].tolist()
    for index, input_text in enumerate(inputs):
        try:
            prompt = f"请根据以下方剂组成，生成 10 个不同表述方式的输入, 每个输入都要包含所有药物，请注意两、斤、升等属于度量单位，不属于药物，同时，小括号内一般为制备方式，请不要修改，请现在开始生成：\n{input_text}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.6,
            )
            # 去除空格和换行符，然后按照数字和点分割, 如 1.第一个元素，2.第二个元素 3.第三个元素。 那么分割后就是 ["第一个元素", "第二个元素", "第三个元素"], 请帮我写出代码
            toSplitText = response.choices[0].message.content.replace("\n", "").replace(" ", "")
            splitTextList = re.split("[\d.]+", toSplitText)
            result = [text for text in splitTextList if text]
            # 将生成的数据添加到原始数据的末尾, 包含 input 和 output 两列
            list_to_append = [{"input": data, "output": outputs[index][5:]} for data in result]
            list_to_append.insert(0,{"input": input_text, "output": outputs[index][5:]})
            pd.DataFrame(list_to_append, columns=["input","output"]).to_csv("../data/data_sets/桂林古本伤寒杂病论药方.train.csv", mode="a", header=False, index=False)
            print('index: ', index,' 生成数据成功')

        except Exception as e:
            print(f"生成数据失败: {e}")

# 读取训练数据
df = pd.read_csv("../data/data_sets/桂林古本伤寒杂病论药方.train.csv")

# 生成增强数据
generate_more_data(df)

# 保存增强数据集
# df.to_csv("../data/data_sets/augmented_train.csv", index=False)

print("增强数据生成完成，已保存到 augmented_train2.csv")