# 导入numpy库，并为其设置别名np  
# 导入pandas库，并为其设置别名pd
import pandas as pd

# 使用pandas的read_csv函数读取名为"creditcard_2023.csv"的CSV文件
# 假设该文件在当前目录下，如果不是，请提供完整的文件路径  
credit_card = pd.read_csv("creditcard_2023.csv")
# 使用DataFrame的tail函数显示数据集的最后几行。默认显示5行。  
# 这里没有指定显示的行数，所以将显示最后5行。  
credit_card.tail()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
credit_card = pd.read_csv("creditcard_2023.csv")
# 绘制直方图
plt.figure(figsize=(9, 4.5))
sns.histplot(credit_card['V1'], bins=25, kde=True, color='darkblue')
plt.title('表1')
# 更改标题为“表1”
plt.xlabel('V1 值')
# 更改x轴标签为“V1 值”
plt.ylabel('频率')
# 更改y轴标签为“频率”
plt.show()
import numpy as np
from sklearn.preprocessing import StandardScaler

# 创建一个随机的二维数组作为示例数据集
x = np.random.rand(100, 2)  # 100个样本，每个样本2个特征
# 创建 StandardScaler 对象    
scaler = StandardScaler()
# 使用 fit_transform 方法对数据进行缩放    
X = scaler.fit_transform(x)
# 打印缩放后的数据    
print(X)
import pandas as pd  # 导入pandas库，用于数据处理和分析  

# 假设 credit_card_data 是一个包含 Amount 和 Class 列的 DataFrame  # 定义一个包含金额和类别两个列的DataFrame
credit_card_data = pd.DataFrame({
    'Amount': [100, 200, 300, 400, 500, 600],  # Amount列的数据  
    'Class': [0, 0, 1, 1, 0, 1]  # Class列的数据，其中0表示正常交易，1表示欺诈交易  
})
# 对credit_card_data进行复制，创建一个新的DataFrame credit_card  
credit_card = credit_card_data.copy()
print("********* Amount Lost due to fraud:************\n")
# 表示欺诈导致的损失金额  
print("Total amount lost to fraud")
# 表示总欺诈损失金额  
print(credit_card.Amount[credit_card.Class == 1].sum())
# 计算所有欺诈交易的金额总和  
print("Mean amount per fraudulent transaction")
# 表示每笔欺诈交易的平均金额  
print(credit_card.Amount[credit_card.Class == 1].mean().round(4))
# 计算所有欺诈交易的平均金额，并保留4位小数  
print("Compare to normal transactions:")
# 表示与正常交易的对比  
print("Total amount from normal transactions")
# 表示正常交易的总金额  
print(credit_card.Amount[credit_card.Class == 0].sum())
# 计算所有正常交易的金额总和  
print("Mean amount per normal transactions")
# 表示每笔正常交易的平均金额  
print(credit_card.Amount[credit_card.Class == 0].mean().round(4))
# 计算所有正常交易的平均金额，并保留4位小数
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
# 使用 seaborn 的 histplot 函数绘制直方图    
plt.figure(figsize=(9, 4.5))
sns.histplot(data['V9'], bins=25, kde=True, color='green')
plt.title('特征分布 V9')
plt.xlabel('V9 值')
plt.ylabel('频率')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
# 指定要读取的CSV文件名  
file_name = 'creditcard_2023.csv'
# 使用pandas库读取CSV文件，并将数据存储在变量data中  
data = pd.read_csv(file_name)
# 创建一个新的图形，大小为9x4.5  
plt.figure(figsize=(9, 4.5))
# 使用seaborn库的histplot函数绘制V17特征的直方图  
sns.histplot(data['V17'], bins=25, kde=True, color='darkblue')
# 设置图形的标题为“特征分布 V17”  
plt.title('特征分布 V17')
# 设置x轴的标签为“V17 值”  
plt.xlabel('V17 值')
# 设置y轴的标签为“频率”  
plt.ylabel('频率')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
# 指定要读取的CSV文件名  
file_name = 'creditcard_2023.csv'
# 使用pandas库读取CSV文件，并将数据存储在变量data中  
data = pd.read_csv(file_name)
# 创建一个新的图形，大小为9x4.5  
plt.figure(figsize=(9, 4.5))
# 使用seaborn库的histplot函数绘制V17特征的直方图  
sns.histplot(data['V26'], bins=25, kde=True, color='darkblue')
# 设置图形的标题为“特征分布 V26”  
plt.title('特征分布 V26')
# 设置x轴的标签为“V26 值”  
plt.xlabel('V26 值')
# 设置y轴的标签为“频率”  
plt.ylabel('频率')
plt.show()
# 导入所需的库  
import pandas as pd
# 导入pandas库，用于数据处理和分析  
import matplotlib.pyplot as plt
# 导入matplotlib库，用于绘图
import seaborn as sns

# 导入seaborn库，基于matplotlib的高级绘图库
# 指定要读取的CSV文件名  
file_name = 'creditcard_2023.csv'
# 使用pandas库读取CSV文件，并将数据存储在变量data中  
data = pd.read_csv(file_name)
# 使用seaborn库绘制数据的核密度估计图  
sns.kdeplot(data=credit_card['Amount'], color='blue', fill=True)
# 绘制金额的核密度估计图，颜色为蓝色，填充图形    
plt.title('金额分配', size=14)
# 显示图形  
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
colors = ['blue', 'green']
explode = [0.1, 0]
credit_card['Class'].value_counts().plot.pie(
    explode=explode,
    autopct='%3.1f%%',
    shadow=True,
    legend=True,
    startangle=45,
    colors=colors,
    wedgeprops=dict(width=0.4)
)
plt.title('类别分布', size=14)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
plt.figure(figsize=(8, 6))
credit_card['Class'].value_counts().plot(kind='bar', color=['blue', 'green'])
plt.title('阶层分布 (0: 非欺诈性, 1: 欺诈性)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['非欺诈性', '欺诈性'])
plt.show()
import pandas as pd

# 读取CSV文件    
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
corrmat = credit_card.corr()
cols = corrmat.nlargest(15, 'Class')['Class'].index
cols
cols_negative = corrmat.nsmallest(15, 'Class')['Class'].index
cols_negative
Credit_card = []
for i in cols:
    Credit_card.append(i)
for j in cols_negative:
    Credit_card.append(j)
Credit_card
x = credit_card.drop(['Class'], axis=1)
y = credit_card.Class
scaler = StandardScaler()
X = scaler.fit_transform(x)
import numpy as np

X = np.random.rand(100, 10)
# 随机生成 100 个样本，每个样本有 10 个特征  
y = np.random.randint(0, 2, 100)
# 随机生成 100 个 0 或 1 的标签
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = SVC()
clf.fit(x_train, y_train)
y_pred_svm = clf.predict(x_train)
y_pred_rf = clf.predict(x_test)
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)

# 假设你有accuracy_values列表来存储每个模型的准确率
accuracy_values = [0.85, 0.80, 0.75, 0.90]  # 示例值，你需要根据实际情况替换这些值  
model_names = ['RandomForest分类器', '支持向量机 ', '逻辑回归', 'xgboost']
bars = plt.bar(model_names, accuracy_values, color=['blue', 'lightblue', 'lightgreen', 'green'])
for bar, value in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')
plt.xlabel('模型')
plt.ylabel('准确')
plt.title('四个模型的准确性')
plt.xticks(rotation=90)
for bar, value in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  # 导入confusion_matrix函数  

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
# 确保y_test和y_pred_rf被定义  
# 例如：  
# y_test = ... # 你的测试目标变量  
# y_pred_rf = ... # 你的随机森林模型的预测输出  
cm_1 = confusion_matrix(y_test, y_pred_rf)
cmn_1 = cm_1.astype('float') / cm_1.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cmn_1, annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  # 导入confusion_matrix函数  

# 读取CSV文件
file_name = 'creditcard_2023.csv'
data = pd.read_csv(file_name)
cm_2 = confusion_matrix(y_test, y_pred_rf)
cmn_2 = cm_2.astype('float') / cm_2.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cmn_2, annot=True, fmt='.2%', cmap='Greens')
plt.ylabel('实际')
plt.xlabel('预测')
plt.show(block=False)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("creditcard_2023.csv")
df.head()
labels = ['0', '1']
sizes = df['Class'].value_counts()
colors = ['Green', 'Red']
explode = (0, 0)
# 创建饼图
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Class Distribution')
# 显示饼图
plt.show()
paper = plt.figure(figsize=[20, 10])
sns.heatmap(df.corr(), cmap='cool', annot=True)
plt.show()
x = df.drop(['Class'], axis=1)
y = df['Class']

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_test = lr.predict(x_test)
y_pred_train = lr.predict(x_train)

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='cividis', linewidths=0.4, square=True, cbar=True,
    xticklabels=["Legit", "Fraud"],
    yticklabels=["Legit", "Fraud"]
)
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.title('混淆矩阵', fontsize=18, fontweight='bold')
plt.yticks(rotation=360)
plt.show()

from sklearn.model_selection import train_test_split