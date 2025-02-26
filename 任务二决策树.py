import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler  # 导入 RandomUnderSampler
from imblearn.over_sampling import SMOTE  # 导入 SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 读取数据
df = pd.read_csv("Heart Disease Dataset.csv")

# 查看数据基本信息
print(df.info())
print(df.describe())

# 处理目标变量（将 'Yes' 转换为 1，'No' 转换为 0）
df["HadHeartAttack"] = df["HadHeartAttack"].map({"Yes": 1, "No": 0})

# 检查是否转换成功
print(df["HadHeartAttack"].value_counts())

# 计算每个 AgeCategory 中 HadHeartAttack == 1 的比例
age_category_proportions = df.groupby('AgeCategory')['HadHeartAttack'].mean().sort_index()
# 绘制折线图
plt.figure(figsize=(16, 6))
plt.plot(age_category_proportions.index, age_category_proportions.values, marker='o', color='b', linestyle='-', markersize=8)
# 设置标题和标签
plt.title('Proportion of Heart Attacks by Age Category', fontsize=14)
plt.xlabel('Age Category', fontsize=6)
plt.ylabel('Proportion of Heart Attacks', fontsize=12)
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# 处理缺失值（这里使用简单填充策略，可根据情况优化）
df.fillna(df.median(numeric_only=True), inplace=True)

# 分离特征和目标变量
X = df.drop(columns=["HadHeartAttack"])  # 预测变量
y = df["HadHeartAttack"]  # 目标变量

# 区分数值型和分类型特征
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 数据预处理：分类变量独热编码
X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)

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

# 使用决策树训练模型，采用CART（Gini指数）作为分裂标准
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    criterion="gini",
    max_depth=20,  # 限制树的深度
    min_samples_split=10,  # 每个节点进行分裂时，至少有10个样本
    min_samples_leaf=5  # 每个叶节点至少包含5个样本
)
dt_classifier.fit(X_resampled, y_resampled)
# 在训练集和测试集上预测
y_train_pred = dt_classifier.predict(X_resampled)  # 训练集预测
y_test_pred = dt_classifier.predict(X_test)  # 测试集预测

# 计算训练集和测试集的准确率
train_accuracy = accuracy_score(y_resampled, y_train_pred)  # 训练集准确率
test_accuracy = accuracy_score(y_test, y_test_pred)  # 测试集准确率

# 输出准确率
print("用于分析特征重要性的决策树在训练集和测试集上的表现：")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 获取特征重要性
feature_importances = dt_classifier.feature_importances_

# 特征名称（包括经过 OneHotEncoder 处理后的分类特征）
feature_names = X_resampled.columns

# 排序特征重要性
sorted_idx = np.argsort(feature_importances)[::-1]

# 绘制特征重要性图，横坐标为特征，纵坐标为特征重要性
plt.figure(figsize=(48, 12))  # 调整图形大小，增加宽度
plt.bar(feature_names[sorted_idx], feature_importances[sorted_idx], align="center")
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Feature Importance from Decision Tree (CART)")
plt.xticks(rotation=90)  # 将x轴标签旋转90度，避免重叠
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.show()
# 获取前 20 个特征及其对应的重要性
top_20_idx = sorted_idx[:20]
top_20_features = feature_names[top_20_idx]
top_20_importances = feature_importances[top_20_idx]
# 绘制前20个特征重要性图
plt.figure(figsize=(12, 8))  # 调整图形大小
plt.barh(top_20_features, top_20_importances, align="center", color='skyblue')  # 使用横向条形图
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title(f"Top 20 Features from Decision Tree (CART)")
plt.tight_layout()  # 自动调整布局
plt.show()
# 输出前 20 个特征及其重要性
print("Top 20 Features by Importance:")
for feature, importance in zip(top_20_features, top_20_importances):
    # 获取特征属于数值型还是分类型
    if feature in num_features:
        feature_type = 'Numeric'
    else:
        feature_type = 'Categorical'

    print(f"Feature: {feature}, Importance: {importance:.4f}, Type: {feature_type}")



binary_features = [
    'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'HighRiskLastYear', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
    'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'DeafOrHardOfHearing'
]

nominal_features = [
    'State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes',
    'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap',
    'CovidPos'
]

numeric_features = [
    'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI'
]


# Plotting the binary features
def plot_binary_features(df, binary_features):
    plt.figure(figsize=(12, 10))
    for idx, feature in enumerate(binary_features):
        plt.subplot(5, 5, idx+1)
        sns.countplot(x=feature, hue='HadHeartAttack', data=df, palette='coolwarm')
        plt.title(f"{feature} vs HadHeartAttack")
        plt.tight_layout()


plot_binary_features(df, binary_features)
plt.show()


def plot_nominal_features(df, nominal_features):
    plt.figure(figsize=(12, 12))
    for idx, feature in enumerate(nominal_features):
        plt.subplot(4, 4, idx+1)
        sns.countplot(x=feature, hue='HadHeartAttack', data=df, palette='coolwarm')
        plt.title(f"{feature} vs HadHeartAttack")
        plt.tight_layout()


plot_nominal_features(df, nominal_features)
plt.show()


def plot_numeric_features(df, numeric_features):
    plt.figure(figsize=(12, 12))
    for idx, feature in enumerate(numeric_features):
        plt.subplot(3, 2, idx+1)
        sns.boxplot(x='HadHeartAttack', y=feature, data=df, palette='coolwarm')
        plt.title(f"{feature} vs HadHeartAttack")
        plt.tight_layout()


plot_numeric_features(df, numeric_features)
plt.show()


'''


# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=feature_names, class_names=["No", "Yes"], filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree Visualization (CART)")
plt.show()
'''

