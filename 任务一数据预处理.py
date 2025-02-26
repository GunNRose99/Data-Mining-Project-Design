import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # 导入 SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler  # 导入 RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# 箱型图检测异常值、异常值处理（数值属性）、类别编码、数据归一化、热力图
df = pd.read_csv('Heart Disease Dataset.csv')
print(df.head())

# 数值属性包括'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI'
num_features = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']


# 类别编码，创建一个 LabelEncoder 实例
label_encoder = LabelEncoder()

# 标称属性
nominal_columns = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes',
                   'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap',
                   'CovidPos']

# 对标称属性进行 LabelEncoder 编码
for col in nominal_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 二元属性
binary_columns = ['BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
                  'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
                  'HighRiskLastYear', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                  'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
                  'DeafOrHardOfHearing']

# 对二元属性进行 LabelEncoder 编码
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 查看编码后的数据
print(df.head())
# 标准化（z-score标准化）
# scaler = StandardScaler()
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 或者使用最小最大归一化
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])
# 异常值处理
# 绘制箱型图
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
plt.tight_layout()
plt.show()
# 使用IQR方法去除异常值
Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的上下限
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 去除异常值
df_cleaned = df[~((df[num_features] < lower_bound) | (df[num_features] > upper_bound)).any(axis=1)]

# 热力图和相关性矩阵
# 计算相关矩阵
correlation_matrix = df.corr()
# 提取与 'HadHeartAttack' 相关的系数（包括 'HadHeartAttack' 自身）
heart_attack_corr = correlation_matrix['HadHeartAttack'].abs()

# 排除 'HadHeartAttack' 自身的相关性（即与自己为 1）
heart_attack_corr = heart_attack_corr.drop('HadHeartAttack')

# 按升序排序并输出结果
heart_attack_corr_sorted = heart_attack_corr.sort_values(ascending=True)

# 打印相关性系数绝对值表格
print("Correlation with 'HadHeartAttack' (absolute values):")
print(heart_attack_corr_sorted)

# 绘制热力图
plt.figure(figsize=(24, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={'size': 6})
plt.title('Correlation Heatmap')
plt.show()
