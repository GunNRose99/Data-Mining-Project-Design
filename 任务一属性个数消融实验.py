# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier  # 导入XGBoost模型
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 读取数据
df = pd.read_csv('Heart Disease Dataset.csv')

# 数值属性
num_features = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']

# 类别编码
label_encoder = LabelEncoder()
nominal_columns = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes',
                   'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap',
                   'CovidPos']

# 对标称属性进行 LabelEncoder 编码
for col in nominal_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 对二元属性进行 LabelEncoder 编码
binary_columns = ['BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
                  'DifficultyDressingBathing',
                  'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
                  'HighRiskLastYear', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                  'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
                  'DeafOrHardOfHearing']

for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 数据归一化
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 计算与 'HadHeartAttack' 的相关性
correlation_matrix = df.corr()
heart_attack_corr = correlation_matrix['HadHeartAttack'].abs().drop('HadHeartAttack')

# 按升序排序
heart_attack_corr_sorted = heart_attack_corr.sort_values(ascending=True)

# 打印相关性
print("Correlation with 'HadHeartAttack' (absolute values):")
print(heart_attack_corr_sorted)

# 删除不同数量的低相关特征
num_deleted_list = [5, 9, 12, 15, 18, 20]

# 设置图形
plt.figure(figsize=(10, 8))

# 用于绘制不同特征组合的 ROC 曲线
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# 遍历不同的删除特征数量
for idx, num_deleted in enumerate(num_deleted_list):
    # 获取最小相关性特征的列
    lowest_corr_columns = heart_attack_corr_sorted.head(num_deleted).index
    df_subset = df.drop(columns=lowest_corr_columns)

    # 特征与目标
    X = df_subset.drop(columns=["HadHeartAttack"])
    y = df_subset["HadHeartAttack"]

    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 使用 TruncatedSVD 进行降维
    svd = TruncatedSVD(n_components=min(len(num_features), int(np.ceil(0.95 * len(X)))) )
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test)

    # SMOTE 过采样
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 欠采样
    under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
    X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)

    # 训练 XGBoost 模型
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_resampled, y_resampled)

    # 预测
    y_val_pred = xgb_model.predict(X_val)
    y_val_prob = xgb_model.predict_proba(X_val)[:, 1]  # 获取预测的概率

    # 计算 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    roc_auc = roc_auc_score(y_val, y_val_prob)

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, lw=2, color=colors[idx], label=f'{num_deleted} features removed (AUC = {roc_auc:.2f})')

    # 输出评估结果
    print(f"Classification Report for {num_deleted} Features Removed:")
    print(classification_report(y_val, y_val_pred))

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print(f"Confusion Matrix for {num_deleted} Features Removed:")
    print(conf_matrix)

# 绘制对角线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Feature Sets')
plt.legend(loc='lower right')
plt.show()
