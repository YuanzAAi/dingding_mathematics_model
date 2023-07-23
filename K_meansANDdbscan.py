# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway, ttest_ind

plt.rcParams["font.family"]="STSong"
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 读取特征数据框
df_cluster_features = pd.read_csv("df_cluster_features.csv")

# 对每个特征进行可视化，画出分布图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.distplot(df_cluster_features["duration"], kde=False)
plt.title("Duration 直方图")
plt.xlabel("Duration")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
sns.distplot(df_cluster_features["up_flow"], kde=False)
plt.title("Up Flow 直方图")
plt.xlabel("Up Flow")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
sns.distplot(df_cluster_features["down_flow"], kde=False)
plt.title("Down Flow 直方图")
plt.xlabel("Down Flow")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('特征数据直方图.png',dpi=300)
plt.show()

# 画出特征数据的核密度图，看看数据是否接近正态分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.kdeplot(df_cluster_features["duration"])
plt.title("Duration 核密度图")
plt.xlabel("Duration")

plt.subplot(1, 3, 2)
sns.kdeplot(df_cluster_features["up_flow"])
plt.title("Up Flow 核密度图")
plt.xlabel("Up Flow")

plt.subplot(1, 3, 3)
sns.kdeplot(df_cluster_features["down_flow"])
plt.title("Down Flow 核密度图")
plt.xlabel("Down Flow")

plt.tight_layout()
plt.savefig('特征数据的核密度图.png',dpi=300)
plt.show()

# 使用对数变换进行处理，使其分布更加平滑
df_cluster_features["duration"] = np.log1p(df_cluster_features["duration"])
df_cluster_features["up_flow"] = np.log1p(df_cluster_features["up_flow"])
df_cluster_features["down_flow"] = np.log1p(df_cluster_features["down_flow"])
df_cluster_features["usage_count"] = np.log1p(df_cluster_features["usage_count"])
df_cluster_features["avg_duration"] = np.log1p(df_cluster_features["avg_duration"])
df_cluster_features["avg_up_flow"] = np.log1p(df_cluster_features["avg_up_flow"])
df_cluster_features["avg_down_flow"] = np.log1p(df_cluster_features["avg_down_flow"])


# 使用归一化对duration进行处理，使其值在0到1之间
#这两个特征的分布非常偏态，有很多零值和极大值，如果直接归一化，会导致数据的变化幅度很小，失去了区分度
scaler = MinMaxScaler()
df_cluster_features["duration"] = scaler.fit_transform(df_cluster_features[["duration"]].values.reshape(-1, 1))
df_cluster_features["up_flow"] = scaler.fit_transform(df_cluster_features["up_flow"].values.reshape(-1, 1))
df_cluster_features["down_flow"] = scaler.fit_transform(df_cluster_features["down_flow"].values.reshape(-1, 1))
df_cluster_features["usage_count"] = scaler.fit_transform(df_cluster_features["usage_count"].values.reshape(-1, 1))
df_cluster_features["avg_duration"] = scaler.fit_transform(df_cluster_features["avg_duration"].values.reshape(-1, 1))
df_cluster_features["avg_up_flow"] = scaler.fit_transform(df_cluster_features["avg_up_flow"].values.reshape(-1, 1))
df_cluster_features["avg_down_flow"] = scaler.fit_transform(df_cluster_features["avg_down_flow"].values.reshape(-1, 1))

# 输出处理后的数据框
print(df_cluster_features)
df_cluster_features.to_csv('df_cluster_features_guiyi.csv', index=False)

# 对每个特征进行可视化，画出分布图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.distplot(df_cluster_features["duration"], kde=False)
plt.title("Duration 直方图")
plt.xlabel("Duration")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
sns.distplot(df_cluster_features["up_flow"], kde=False)
plt.title("Up Flow 直方图")
plt.xlabel("Up Flow")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
sns.distplot(df_cluster_features["down_flow"], kde=False)
plt.title("Down Flow 直方图")
plt.xlabel("Down Flow")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('特征数据处理后直方图.png',dpi=300)
plt.show()

# 画出特征数据的核密度图，看看数据是否接近正态分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.kdeplot(df_cluster_features["duration"])
plt.title("Duration 核密度图")
plt.xlabel("Duration")

plt.subplot(1, 3, 2)
sns.kdeplot(df_cluster_features["up_flow"])
plt.title("Up Flow 核密度图")
plt.xlabel("Up Flow")

plt.subplot(1, 3, 3)
sns.kdeplot(df_cluster_features["down_flow"])
plt.title("Down Flow 核密度图")
plt.xlabel("Down Flow")

plt.tight_layout()
plt.savefig('特征数据处理后核密度图.png',dpi=300)
plt.show()

# 画出箱线图可视化
plt.figure(figsize=(12, 4))
sns.boxplot(data=df_cluster_features[["duration", "up_flow", "down_flow"]])
plt.title("特征箱线图（数据处理后）")
plt.savefig('特征数据处理后箱线图.png',dpi=300)
plt.show()

# 定义一个函数，计算不同K值下的SSE值
def get_sse(data, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    return sse

# 使用“手肘法”确定K值，可视化SSE值随K值的变化
k_range = range(2, 11)
sse = get_sse(df_cluster_features[["duration", "up_flow", "down_flow"]], k_range)
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("手肘法找最佳K值")
plt.savefig('手肘法.png',dpi=300)
plt.show()

# 使用K-means聚类，K=4
kmeans = KMeans(n_clusters=4,max_iter=10000, init="k-means++", tol=1e-6,random_state=0)
kmeans.fit(df_cluster_features[["duration", "up_flow", "down_flow"]])
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 创建一个新的数据框，存储聚类结果
df_kmeans_cluster_result = pd.DataFrame({"user_id": df_cluster_features["user_id"], "cluster": labels})

# 输出新的数据框
print(df_kmeans_cluster_result)
df_kmeans_cluster_result.to_csv('df_kmeans_cluster_result.csv', index=False)

# 对聚类结果进行可视化，使用三维散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
# 定义一个颜色列表，用于表示不同类别用户
colors = ["red", "skyblue", "blue", "yellow"]
# 定义一个标签列表，用于表示不同类别用户的名称
labels = ["0", "1", "2", "3"]
# 使用循环来绘制每个类别用户的散点图，并添加图例
for i in range(4):
    ax.scatter(df_cluster_features[df_kmeans_cluster_result["cluster"] == i]["duration"], df_cluster_features[df_kmeans_cluster_result["cluster"] == i]["up_flow"], df_cluster_features[df_kmeans_cluster_result["cluster"] == i]["down_flow"], c=colors[i], label=labels[i])
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="black", marker="x", s=100)
ax.set_xlabel("Duration")
ax.set_ylabel("Up Flow")
ax.set_zlabel("Down Flow")
ax.set_title("手肘法-K均值聚类图")
# 添加图例
ax.legend()
plt.savefig('手肘法-K均值聚类图.png',dpi=300)
plt.show()


# 使用PCA降维，将高维数据映射到二维平面上
pca = PCA(n_components=2, random_state=0)
pca.fit(df_cluster_features[["duration", "up_flow", "down_flow","usage_count","avg_duration","avg_up_flow","avg_down_flow"]])
data_pca = pca.transform(df_cluster_features[["duration", "up_flow", "down_flow","usage_count","avg_duration","avg_up_flow","avg_down_flow"]])
# 将data_pca转换为DataFrame
df_pca = pd.DataFrame(data_pca, columns=["pca1", "pca2"])
df_pca.to_csv('df_pca.csv', index=False)

# 使用K-means聚类，K=4
kmeans = KMeans(n_clusters=4,max_iter=10000, init="k-means++", tol=1e-6,random_state=0)
kmeans.fit(df_pca)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 创建一个新的数据框，存储聚类结果
df_pca_kmeans_cluster_result = pd.DataFrame({"user_id": df_cluster_features["user_id"], "cluster": labels,
                                             "pca1": df_pca["pca1"],
                                             "pca2": df_pca["pca2"]})
print(df_pca_kmeans_cluster_result)

# 对降维后的数据进行可视化，使用二维散点图
# 使用循环来绘制每个类别用户的散点图，并添加图例
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)  # 获取唯一的类别值
# 使用循环来绘制每个类别用户的散点图，并添加图例
for i in range(4):
    mapped_label = np.where(unique_labels == i)[0][0]  # 将类别映射为 0、1、2、3
    plt.scatter(df_pca[df_pca_kmeans_cluster_result["cluster"] == i]['pca1'], df_pca[df_pca_kmeans_cluster_result["cluster"] == i]['pca2'], c=colors[i], label=mapped_label)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA降维-K均值聚类图")
# 添加图例
plt.legend()
plt.savefig('PCA降维-K均值聚类图.png',dpi=300)
plt.show()


# 使用ANOVA和t-test来评估聚类效果和意义
# 定义一个函数，计算不同类别用户在某个指标上的均值和方差
def get_mean_var(data, cluster, feature):
    mean_var = []
    for i in range(cluster):
        mean_var.append([data[data["cluster"] == i][feature].mean(), data[data["cluster"] == i][feature].var()])
    return mean_var

# 定义一个函数，计算不同类别用户在某个指标上的ANOVA和t-test结果
def get_anova_ttest(data, cluster, feature):
    anova_ttest = []
    for i in range(cluster):
        for j in range(i + 1, cluster):
            anova_ttest.append([i, j, f_oneway(data[data["cluster"] == i][feature], data[data["cluster"] == j][feature])[1], ttest_ind(data[data["cluster"] == i][feature], data[data["cluster"] == j][feature])[1]])
    return anova_ttest

# 定义一个函数，将结果输出到一个新的数据框，并保存到文件
def output_result(data, cluster, feature):
    # 计算均值和方差
    mean_var = get_mean_var(data, cluster, feature)
    # 计算ANOVA和t-test
    anova_ttest = get_anova_ttest(data, cluster, feature)
    # 创建一个新的数据框，存储结果
    df_result = pd.DataFrame(columns=["cluster1", "cluster2", "mean1", "var1", "mean2", "var2", "anova_pvalue", "ttest_pvalue"])
    for i in range(len(anova_ttest)):
        df_result.loc[i] = [anova_ttest[i][0], anova_ttest[i][1], mean_var[anova_ttest[i][0]][0], mean_var[anova_ttest[i][0]][1], mean_var[anova_ttest[i][1]][0], mean_var[anova_ttest[i][1]][1], anova_ttest[i][2], anova_ttest[i][3]]
    # 输出新的数据框
    print(df_result)
    df_result.to_csv('df_ANOVA_' + feature + '.csv', index=False)

# 将df_kmeans_cluster_result和df_cluster_features合并成一个新的数据框
df_merged = pd.merge(df_kmeans_cluster_result, df_cluster_features, on="user_id")

# 对使用时长，上行流量，下行流量三个指标分别进行评估
#通过查看cluster1和cluster2两列的值，来知道每一行是哪个跟哪个比。
#例如，第一行的cluster1是0，cluster2是1，表示这一行是cluster0和cluster1的比较
output_result(df_merged, 4, "duration")
output_result(df_merged, 4, "up_flow")
output_result(df_merged, 4, "down_flow")

'''dbscan'''
'''调参'''
# 定义eps和min_samples的参数范围
eps_range = np.arange(0.01,0.1,0.01)
min_samples_range = np.arange(4, 10, 1)

# 创建一个空的数据框，存储评价指标
df_evaluation = pd.DataFrame(columns=["eps", "min_samples", "silhouette_score", "noise_ratio"])

# 使用循环来尝试不同的参数组合，并计算轮廓系数和噪声比
for eps in eps_range:
    for min_samples in min_samples_range:
        # 使用DBSCAN密度聚类算法进行聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples,n_jobs=-1)
        dbscan.fit(df_cluster_features[["duration", "up_flow", "down_flow"]])
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 计算聚类的个数，排除噪声点
        n_noise = list(labels).count(-1)  # 计算噪声点的个数

        # 如果聚类个数大于1，就计算轮廓系数，否则设为-1
        if n_clusters > 1:
            silhouette_score = metrics.silhouette_score(df_cluster_features[["duration", "up_flow", "down_flow"]], labels)  # 计算轮廓系数
        else:
            silhouette_score = -1

        # 计算噪声比，即噪声点个数占总数的比例
        noise_ratio = (n_noise / len(labels))*1000

        # 将结果添加到数据框中
        df_evaluation.loc[len(df_evaluation)] = [eps, min_samples, silhouette_score, noise_ratio]

# 输出数据框
print(df_evaluation)
df_evaluation.to_csv('df_evaluation.csv', index=False)

# 对评价指标进行可视化，使用散点图
sns.relplot(x="eps",y="min_samples", size='silhouette_score',data=df_evaluation)
plt.title("轮廓系数散点图")
plt.savefig('DBSCAN调参轮廓系数散点图.png', dpi=300)

sns.relplot(x="eps",y="min_samples", size='noise_ratio',data=df_evaluation)
plt.title("噪声比散点图")
plt.savefig('DBSCAN调参噪声比散点图.png', dpi=300)
# 使用DBSCAN密度聚类算法进行聚类，epsilon（ɛ）取值为0.09，minPts取值为4
dbscan = DBSCAN(eps=0.09, min_samples=4,n_jobs=-1)
dbscan.fit(df_cluster_features[["duration", "up_flow", "down_flow"]])
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # 计算聚类的个数，排除噪声点

# 创建一个新的数据框，存储聚类结果
df_dbscan_cluster_result = pd.DataFrame({"user_id": df_cluster_features["user_id"], "cluster": labels})

# 输出新的数据框
print(df_dbscan_cluster_result)
df_dbscan_cluster_result.to_csv('df_dbscan_cluster_result.csv', index=False)

# 对聚类结果进行可视化，使用三维散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
# 定义一个颜色列表，用于表示不同类别用户
colors = ["red", "skyblue", "blue", "yellow", "cyan", "magenta", "black"]
# 定义一个标签列表，用于表示不同类别用户的名称
labels = ["0", "1", "2", "3", "4", "5", "noise"]
# 使用循环来绘制每个类别用户的散点图，并添加图例
for i in range(-1, n_clusters ):
    ax.scatter(df_cluster_features[df_dbscan_cluster_result["cluster"] == i]["duration"], df_cluster_features[df_dbscan_cluster_result["cluster"] == i]["up_flow"], df_cluster_features[df_dbscan_cluster_result["cluster"] == i]["down_flow"], c=colors[i], label=labels[i])
ax.set_xlabel("Duration")
ax.set_ylabel("Up Flow")
ax.set_zlabel("Down Flow")
ax.set_title("DBSCAN密度聚类图")
# 添加图例
ax.legend()
plt.savefig('DBSCAN密度聚类图.png', dpi=300)
plt.show()


'''调参'''
# 创建一个空的数据框，存储评价指标
df_pca_evaluation = pd.DataFrame(columns=["eps", "min_samples", "silhouette_score", "noise_ratio"])

# 使用循环来尝试不同的参数组合，并计算轮廓系数和噪声比
for eps in eps_range:
    for min_samples in min_samples_range:
        # 使用DBSCAN密度聚类算法进行聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples,n_jobs=-1)
        dbscan.fit(data_pca)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 计算聚类的个数，排除噪声点
        n_noise = list(labels).count(-1)  # 计算噪声点的个数

        # 如果聚类个数大于1，就计算轮廓系数，否则设为-1
        if n_clusters > 1:
            silhouette_score = metrics.silhouette_score(data_pca, labels)  # 计算轮廓系数
        else:
            silhouette_score = -1

        # 计算噪声比，即噪声点个数占总数的比例
        noise_ratio = (n_noise / len(labels))*1000

        # 将结果添加到数据框中
        df_pca_evaluation.loc[len(df_pca_evaluation)] = [eps, min_samples, silhouette_score, noise_ratio]

# 输出数据框
print(df_pca_evaluation)
df_pca_evaluation.to_csv('df_pca_evaluation.csv', index=False)

# 对评价指标进行可视化，使用散点图
sns.relplot(x="eps",y="min_samples", size='silhouette_score',data=df_pca_evaluation)
plt.title("轮廓系数散点图")
plt.savefig('pca-DBSCAN调参轮廓系数散点图.png', dpi=300)

sns.relplot(x="eps",y="min_samples", size='noise_ratio',data=df_pca_evaluation)
plt.title("噪声比散点图")
plt.savefig('pca-DBSCAN调参噪声比散点图.png', dpi=300)

# 使用pca-DBSCAN密度聚类算法进行聚类，epsilon（ɛ）取值为0.09，minPts取值为4
dbscan = DBSCAN(eps=0.09, min_samples=4,n_jobs=-1)
dbscan.fit(df_pca)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # 计算聚类的个数，排除噪声点
# 创建一个新的数据框，存储聚类结果
df_pca_dbscan_cluster_result = pd.DataFrame({"user_id": df_cluster_features["user_id"], "cluster": labels,
                                            "pca1": df_pca["pca1"],"pca2": df_pca["pca2"]})

# 对降维后的数据进行可视化，使用二维散点图
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)  # 获取唯一的类别值
# 使用循环来绘制每个类别用户的散点图，并添加图例
for i in range(-1, n_clusters):
    mapped_label = np.where(unique_labels == i)[0][0]  # 将类别映射为 0、1、2、3
    plt.scatter(df_pca[df_pca_dbscan_cluster_result["cluster"] == i]['pca1'], df_pca[df_pca_dbscan_cluster_result["cluster"] == i]['pca2'], c=colors[i], label=mapped_label)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA降维-DBSCAN密度聚类图")
# 添加图例
plt.legend()
plt.savefig('PCA降维-DBSCAN密度聚类图.png',dpi=300)
plt.show()
