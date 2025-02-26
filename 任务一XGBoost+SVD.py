import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE  # 导入 SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier  # 导入 XGBoost 分类器

# 读取数据
df = pd.read_csv("Heart Disease Dataset.csv")

# 查看数据基本信息
print(df.info())
print(df.describe())

# 处理目标变量（将 'Yes' 转换为 1，'No' 转换为 0）
df["HadHeartAttack"] = df["HadHeartAttack"].map({"Yes": 1, "No": 0})

# 检查是否转换成功
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
# 1. 数值特征标准化
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# 2. 分类特征独热编码
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X[cat_features])

# 将独热编码的分类特征与标准化后的数值特征结合
X_preprocessed = np.hstack((X[num_features].values, X_encoded))

# 计算保留95%方差所需的 TruncatedSVD 组件数
n_components = min(len(num_features), int(np.ceil(0.95 * len(X))))  # 设置为一个合理的整数

# 使用 TruncatedSVD 进行降维
svd = TruncatedSVD(n_components=n_components)
X_svd = svd.fit_transform(X_preprocessed)

# 划分数据集（80% 训练集，20% 测试集）
X_train, X_temp, y_train, y_temp = train_test_split(X_svd, y, test_size=0.2, random_state=42, stratify=y)

# 进一步划分出验证集（10% 验证集，10% 测试集）
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 解决类别不平衡 - 过采样（SMOTE）
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # 使用 SMOTE 进行过采样
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # 使用欠采样
X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)

# 创建 XGBoost 分类器
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1)  # scale_pos_weight 用于类别不平衡

# 训练模型
xgb_classifier.fit(X_resampled, y_resampled)

# 预测
y_pred_xgb = xgb_classifier.predict(X_test)

# 计算并打印模型评估指标
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
f1_xgb_overall = f1_score(y_test, y_pred_xgb, average='weighted')
print(f"XGBoost Overall F1 Score: {f1_xgb_overall:.4f}")

# 绘制混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.show()

# 计算 AUC-ROC
y_prob_xgb = xgb_classifier.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)

print(f"XGBoost AUC-ROC: {roc_auc_xgb:.4f}")

# 绘制 ROC 曲线
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()
