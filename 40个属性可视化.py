import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('Heart Disease Dataset.csv')

# 查看数据集的前几行
print(df.head())

# 查看缺失值情况
print(df.isnull().sum())

selected_features_1 = [
    'State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays',
    'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities',
    'SleepHours', 'RemovedTeeth', 'HadHeartAttack'
]

df_selected_1 = df[selected_features_1]
sns.set(style="whitegrid")

# 可视化第一个特征集
fig1, axes1 = plt.subplots(5, 2, figsize=(14, 25))
for i, feature in enumerate(selected_features_1):
    ax = axes1[i // 2, i % 2]  # 按行列位置选择子图
    if df[feature].dtype == 'object' or len(df[feature].unique()) < 10:  # 类别变量
        sns.countplot(x=feature, data=df, ax=ax)
        ax.tick_params(axis='x', rotation=45)
    else:  # 数值型变量
        sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'{feature} Distribution')

# 调整布局
fig1.tight_layout()
fig1.suptitle('First 10 Features', fontsize=16, y=1.03)

# 显示第一个画布
plt.show()


selected_features_2 = [
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
    'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing'
]

df_selected_2 = df[selected_features_2]
sns.set(style="whitegrid")

# 可视化第二个特征集
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(selected_features_2):
    ax = axes2[i // 5, i % 5]  # 按行列位置选择子图
    if df[feature].dtype == 'object' or len(df[feature].unique()) < 10:
        sns.countplot(x=feature, data=df, ax=ax)
        ax.tick_params(axis='x', rotation=45)
    else:  # 数值型变量
        sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'{feature} Distribution')

# 调整布局
fig2.tight_layout()
fig2.suptitle('Next 10 Features', fontsize=16, y=1.03)

# 显示第二个画布
plt.show()


selected_features_3 = [
    'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
    'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
    'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory'
]

df_selected_3 = df[selected_features_3]
sns.set(style="whitegrid")

fig3, axes3 = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(selected_features_3):
    ax = axes3[i // 5, i % 5]  # 按行列位置选择子图
    if df[feature].dtype == 'object' or len(df[feature].unique()) < 10:  # 类别变量
        sns.countplot(x=feature, data=df, ax=ax)
        ax.tick_params(axis='x', rotation=45)
    else:  # 数值型变量
        sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'{feature} Distribution')

# 调整布局
fig3.tight_layout()
fig3.suptitle('Features: Blind, Difficulty, Smoker, Race & Age', fontsize=16, y=1.03)  # 添加标题

# 显示图形
plt.show()


selected_features_4 = [
    'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers',
    'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
    'HighRiskLastYear', 'CovidPos'
]

df_selected_4 = df[selected_features_4]
sns.set(style="whitegrid")

fig4, axes4 = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(selected_features_4):
    ax = axes4[i // 5, i % 5]  # 按行列位置选择子图
    if df[feature].dtype == 'object' or len(df[feature].unique()) < 10:  # 类别变量
        sns.countplot(x=feature, data=df, ax=ax)
        ax.tick_params(axis='x', rotation=45)
    else:  # 数值型变量
        sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'{feature} Distribution')

# 调整布局
fig4.tight_layout()
fig4.suptitle('Health and Risk Features', fontsize=16, y=1.03)

# 显示图形
plt.show()