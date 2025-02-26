import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # 导入 SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler  # 导入 RandomUnderSampler
from xgboost import XGBClassifier  # 导入 XGBoost 分类器

# 读取数据
df = pd.read_csv('Heart Disease Dataset.csv')
print(df.head())

# 数值属性
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

# 最小最大归一化
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

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

# 计算与 'HadHeartAttack' 相关系数最小的9个属性
lowest_corr_columns = heart_attack_corr_sorted.head(9).index

# 删除这些属性
df = df.drop(columns=lowest_corr_columns)
X = df.drop(columns=["HadHeartAttack"])  # 预测变量
y = df["HadHeartAttack"]  # 目标变量

# 打印删除后的数据框
print("DataFrame after dropping least correlated features with 'HadHeartAttack':")
print(df.head())

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 使用 TruncatedSVD 进行降维
svd = TruncatedSVD(n_components=min(len(num_features), int(np.ceil(0.95 * len(X)))) )  # 设置为一个合理的整数
X_train_svd = svd.fit_transform(X_train)
X_val_svd = svd.transform(X_val)
X_test_svd = svd.transform(X_test)

# 采样
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # 先进行过采样
# 打印过采样后的目标变量分布
print("Target distribution after SMOTE:")
print(y_resampled.value_counts())

under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)  # 然后进行欠采样
# 打印过采样和欠采样后的目标变量分布
print("Target distribution after SMOTE and RandomUnderSampling:")
print(y_resampled.value_counts())

# 使用 XGBoost 进行训练
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1)

# 训练模型
xgb_classifier.fit(X_resampled, y_resampled)

# 预测
y_pred_xgb = xgb_classifier.predict(X_test)

# 计算评估指标
print("XGBoost Evaluation Metrics:")

# 打印准确率
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

# 打印分类报告
print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb)}")

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred_xgb)
print(f"Confusion Matrix:\n{cm}")

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Heart Attack", "Heart Attack"], yticklabels=["No Heart Attack", "Heart Attack"])
plt.title("Confusion Matrix - XGBoost")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
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
plt.tight_layout()
plt.show()
