import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("Heart Disease Dataset.csv")

# 去掉 "ChestScan" 列，并更新 binary_columns
binary_columns = ["HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
                  "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
                  "HadArthritis", "HadDiabetes", "HadHeartAttack"]

# 筛选出指定的列
df = df[binary_columns]

# 将 "Yes" 转换为 1，"No" 转换为 0
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# 计算每一行中有多少个属性值为 1（不包括 HadHeartAttack）
df['num_ones'] = df[binary_columns[:-1]].sum(axis=1)

# 计算每个 num_ones 对应下 HadHeartAttack 为 1 的概率
grouped = df.groupby('num_ones')['HadHeartAttack'].mean()

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(grouped.index, grouped.values, color='skyblue')
plt.title('Probability of HadHeartAttack=1 for Different Numbers of 1s in Binary Attributes')
plt.xlabel('Number of 1s in Binary Attributes')
plt.ylabel('Probability of HadHeartAttack=1')
plt.xticks(range(10))  # 横坐标范围为0到9
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
