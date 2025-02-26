import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD

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
binary_columns = ['BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
                  'DifficultyDressingBathing',
                  'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
                  'HighRiskLastYear', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                  'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
                  'DeafOrHardOfHearing']

# 对二元属性进行 LabelEncoder 编码
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 最小最大归一化
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 计算相关矩阵
correlation_matrix = df.corr()
heart_attack_corr = correlation_matrix['HadHeartAttack'].abs()
heart_attack_corr = heart_attack_corr.drop('HadHeartAttack')
heart_attack_corr_sorted = heart_attack_corr.sort_values(ascending=True)

# 删除与 'HadHeartAttack' 相关系数最小的9个属性
lowest_corr_columns = heart_attack_corr_sorted.head(9).index
df = df.drop(columns=lowest_corr_columns)
X = df.drop(columns=["HadHeartAttack"])
y = df["HadHeartAttack"]

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 使用 TruncatedSVD 进行降维
svd = TruncatedSVD(n_components=min(len(num_features), int(np.ceil(0.95 * len(X)))))  # 设置为一个合理的整数
X_train_svd = svd.fit_transform(X_train)
X_val_svd = svd.transform(X_val)
X_test_svd = svd.transform(X_test)

# 采样
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_svd, y_train)  # 先进行过采样
under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)  # 然后进行欠采样

# 定义分类器和调参网格
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42),
        "params": {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    },
    "XGBoost": {
        "model": XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1),
        "params": {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    }
}

# 存储 ROC 数据
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))

# 对每个模型进行网格搜索调参，训练、预测和评估
for model_name, model_info in models.items():
    model = model_info["model"]
    params = model_info["params"]

    # 网格搜索调参
    grid_search = GridSearchCV(model, params, scoring='f1', cv=3, n_jobs=1)
    grid_search.fit(X_resampled, y_resampled)

    # 获取最优模型
    best_model = grid_search.best_estimator_

    # 预测
    y_prob = best_model.predict_proba(X_test_svd)[:, 1]  # 获取正类的预测概率
    y_pred = best_model.predict(X_test_svd)

    # 计算 AUC-ROC
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_dict[model_name] = fpr
    tpr_dict[model_name] = tpr
    roc_auc_dict[model_name] = roc_auc

    # 绘制每个模型的 ROC 曲线（不同颜色区分）
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

# 绘制随机猜测线
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 打印每个模型的评估指标
for model_name, model_info in models.items():
    model = model_info["model"]
    params = model_info["params"]

    # 网格搜索调参
    grid_search = GridSearchCV(model, params, scoring='f1', cv=3, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    # 获取最优模型
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_svd)

    print(f"\n{model_name} Evaluation Metrics (Best Model):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Heart Attack", "Heart Attack"],
                yticklabels=["No Heart Attack", "Heart Attack"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
