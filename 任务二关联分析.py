import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 读取数据集
df = pd.read_csv('Heart Disease Dataset.csv')

# 数值属性列
num_features = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']
# 对数值属性列进行基于均值的划分 (0 和 1)
for col in num_features:
    mean_value = df[col].mean()  # 计算列的均值
    df[col] = df[col].apply(lambda x: 1 if x >= mean_value else 0)

# 输出处理后的数据
print("DataFrame after applying mean-based binarization:")
print(df[num_features].head())

# 类别编码，创建一个 LabelEncoder 实例
label_encoder = LabelEncoder()

# 类别列
nominal_columns = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes',
                   'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap',
                   'CovidPos']

# 对标称属性进行 LabelEncoder 编码
for col in nominal_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 对标称属性列进行 One-Hot 编码
df_encoded = pd.get_dummies(df, columns=nominal_columns)

# 输出编码后的数据
print("One-Hot Encoded DataFrame:")
print(df_encoded.head())

# 二元属性列
binary_columns = ['BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
                  'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
                  'HighRiskLastYear', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                  'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
                  'DeafOrHardOfHearing']

# 对二元属性进行 LabelEncoder 编码
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 将二元属性转换为布尔值(0/1)
for col in binary_columns:
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

# 确保所有列为布尔类型（0 或 1），但只处理数值类型的列
df_encoded = df_encoded.applymap(lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)

# 使用FP-Growth进行频繁项集挖掘
frequent_itemsets = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)

# 输出频繁项集
print("Frequent Itemsets:")
print(frequent_itemsets)

# 挖掘关联规则
association_rules_result = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)

# 输出关联规则并显示支持度、置信度、提升度
print("\nAssociation Rules with Support, Confidence, and Lift:")
print(association_rules_result[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\nAssociation Rules with Support, Confidence, and Lift:")
# 筛选关联规则，后件为 HadHeartAttack_0 或 HadHeartAttack_1
filtered_rules = association_rules_result[association_rules_result['consequents'].apply(lambda x: 'HadHeartAttack_0' in str(x) or 'HadHeartAttack_1' in str(x))]
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))