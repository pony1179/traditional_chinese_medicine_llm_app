import re
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_formulas(text):
    formulas = []  # 初始化一个空列表，用于存储所有解析出的方剂信息
    current = {}  # 初始化一个空字典，用于存储当前正在解析的方剂信息
    
    for line in text.split('\n'):  # 遍历文本中的每一行
        line = line.strip()  # 去除行首行尾的空白字符
        
        # 识别新方剂
        if re.match(r'^\d+\.', line):  # 如果行以数字和点开头，表示是一个新的方剂
            if current:  # 如果当前方剂信息非空，则将其添加到方剂列表中
                formulas.append(current)
                current = {}  # 重置当前方剂信息
            parts = line.split('方', 1)  # 将行按'方'分割，获取方剂名称和后续内容
            current["名称"] = parts[0].split('.',1)[1].strip() + "方"  # 提取方剂名称
            current["组成"] = []  # 初始化方剂组成列表
            current["制备"] = []  # 初始化方剂制备方法列表
            continue  # 跳过当前循环，处理下一行
            
        # 提取组成
        if '右' not in line and '方' not in line and line:  # 如果行不包含'右'和'方'，且非空
            if '以水' not in line:  # 排除制备步骤
                ingredients = re.split(r'\s{2,}', line)  # 按多个空格分割行，获取组成成分
                current["组成"].extend([i.strip() for i in ingredients if i.strip()])  # 去除空白并添加到组成列表
            
        # 提取制备方法
        if line.startswith('右') or '以水' in line:  # 如果行以'右'开头或包含'以水'，表示是制备方法
            current["制备"].append(line)

    if current:
        formulas.append(current)
    return formulas

# 读取文本文件
with open('../data/extracted_books/桂林古本伤寒杂病论药方.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 解析数据
formulas_data = parse_formulas(text)

# 转换为DataFrame
df = pd.DataFrame(formulas_data)

# 合并字段
df['组成'] = df['组成'].apply(lambda x: '\n'.join(x))
df['制备'] = df['制备'].apply(lambda x: '\n'.join(x))

# 保存CSV
df.to_csv('../data/data_sets/桂林古本伤寒杂病论药方.csv', index=False, encoding='utf-8-sig')


# 创建问答格式示例
df['input'] = "方剂组成：" + df['组成'] + "\n制备方法：" + df['制备']
df['output'] = "方剂名称：" + df['名称']

# 分割数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存为Hugging Face数据集格式
train_df[['input', 'output']].to_csv('../data/data_sets/桂林古本伤寒杂病论药方.train.csv', index=False)
val_df[['input', 'output']].to_csv('../data/data_sets/桂林古本伤寒杂病论药方.val.csv', index=False)