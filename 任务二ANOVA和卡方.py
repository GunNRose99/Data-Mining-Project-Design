import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.under_sampling import RandomUnderSampler  # 导入 RandomUnderSampler

# 读取数据
df = pd.read_csv("Heart Disease Dataset.csv")

# 查看数据基本信息
print(df.info())
print(df.describe())

# 处理目标变量（将 'Yes' 转换为 1，'No' 转换为 0）
df["HadHeartAttack"] = df["HadHeartAttack"].map({"Yes": 1, "No": 0})

# 检查是否转换成功
print("Before SMOTE:")
print(df["HadHeartAttack"].value_counts())

# 处理缺失值（这里使用简单填充策略，可根据情况优化）
df.fillna(df.median(numeric_only=True), inplace=True)

# 分离特征和目标变量
X = df.drop(columns=["HadHeartAttack"])  # 预测变量
y = df["HadHeartAttack"]  # 目标变量

# 区分数值型和分类型特征
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 数据预处理：分类变量独热编码，数值变量标准化
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_features),  # 数值特征归一化
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),  # 分类特征独热编码
    ]
)

# 先进行独热编码，将分类变量转换为数值型
X_encoded = preprocessor.fit_transform(X)

# Get the feature names after encoding
encoded_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(cat_features)

# Combine the numerical features and encoded categorical features' names
all_feature_names = num_features + list(encoded_feature_names)

# 先按 7:1.5:1.5 划分数据集为训练集、测试集、验证集
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)  # 70% 训练集，30% 测试+验证集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 15% 验证集，15% 测试集


# 打印每个数据集的类别分布
print("Training set class distribution:")
print(y_train.value_counts())

print("\nValidation set class distribution:")
print(y_val.value_counts())

print("\nTest set class distribution:")
print(y_test.value_counts())

# 在训练集上进行欠采样和过采样
# 先进行欠采样再过采样
under_sampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42)  # 使用欠采样
X_resampled, y_resampled = under_sampler.fit_resample(X_train, y_train)

# 打印欠采样后的类别数据数量
print(f"After Random Under-sampling, the class distribution in the training set is:\n{y_resampled.value_counts()}")

# 使用SMOTE进行过采样
smote = SMOTE(sampling_strategy='auto', random_state=42)  # 使用 SMOTE 进行过采样
X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

# 打印过采样后的类别数据数量
print(f"After SMOTE Over-sampling, the class distribution in the training set is:\n{y_resampled.value_counts()}")

# 对非数值型特征进行卡方检验（Chi-Squared）
# 只选择分类型特征进行卡方检验
X_cat_resampled = X_resampled[:, len(num_features):]  # 只提取经过编码后的分类特征

chi2_selector = SelectKBest(chi2, k='all')
X_new_1 = chi2_selector.fit_transform(X_cat_resampled, y_resampled)

# 获取 p 值和特征分数
chi2_scores = pd.DataFrame(data=chi2_selector.scores_, index=encoded_feature_names, columns=['Chi Squared Score'])

# 可视化Chi Squared 结果
plt.figure(figsize=(10, 10))  # Increase figure size to avoid overcrowding
sns.heatmap(chi2_scores.sort_values(ascending=False, by='Chi Squared Score').head(20),  # Display top 20 features
            annot=True,
            cmap='coolwarm',
            linewidths=0.4,
            linecolor='black',
            fmt='.2f',
            cbar_kws={'label': 'Chi Squared Score'})  # Adding color bar label
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
plt.title('Top 20 Chi-Squared Feature Selection for Categorical Features')  # Title for clarity
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()


# 对数值型特征进行ANOVA检验（f_classif）
# 只选择数值型特征进行ANOVA检验
X_num_resampled = X_resampled[:, :len(num_features)]  # 只提取数值特征

anova_selector = SelectKBest(f_classif, k='all')
X_new_2 = anova_selector.fit_transform(X_num_resampled, y_resampled)

# 获取 p 值和特征分数
anova_scores = pd.DataFrame(data=anova_selector.scores_, index=num_features, columns=['ANOVA Score'])

# 可视化 ANOVA 结果
plt.figure(figsize=(10, 10))  # Increase figure size to avoid overcrowding
sns.heatmap(anova_scores.sort_values(ascending=False, by='ANOVA Score').head(20),  # Display top 20 features
            annot=True,
            cmap='coolwarm',
            linewidths=0.4,
            linecolor='black',
            fmt='.2f',
            cbar_kws={'label': 'ANOVA Score'})  # Adding color bar label
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
plt.title('Top 20 ANOVA Feature Selection for Numerical Features')  # Title for clarity
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()
