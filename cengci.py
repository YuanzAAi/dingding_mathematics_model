import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import metrics
plt.rcParams["font.family"]="STSong"
pd.options.display.float_format = '{:.2f}'.format


# 读取特征数据框
df_cluster_features = pd.read_csv("df_cluster_features_guiyi.csv")
df_pca = pd.read_csv("df_pca.csv")

'''hierarchical clustering'''
# 定义两种数据集，一种是原始数据，一种是PCA降维后的数据
datasets = {"原始数据": df_cluster_features[["duration", "up_flow", "down_flow"]],"PCA降维后的数据": df_pca}

# 使用循环来对两种数据集分别进行层次聚类
for name, data in datasets.items():
    '''调参'''
    # 定义t值的参数范围
    t_range = np.arange(2, 6, 1)

    # 创建一个空的数据框，存储评价指标
    df_hc_evaluation = pd.DataFrame(columns=["t", "silhouette_score"])

    # 使用循环来尝试不同的t值，并计算轮廓系数
    for t in t_range:
        try:
            # 使用层次聚类算法进行聚类（使用平均距离法）
            Z = linkage(data, method="average")
            labels = fcluster(Z, t, criterion="maxclust")

            # 计算轮廓系数
            silhouette_score = metrics.silhouette_score(data, labels)
            # 将结果添加到数据框中
            df_hc_evaluation.loc[len(df_hc_evaluation)] = [t, silhouette_score]
        except ValueError:
            silhouette_score = 0

    # 输出数据框
    print(name + "层次聚类调参结果：")
    print(df_hc_evaluation)
    df_hc_evaluation.to_csv('df_hc_evaluation_' + name + '.csv', index=False)

    # 对评价指标进行可视化，使用折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df_hc_evaluation["t"], df_hc_evaluation["silhouette_score"])
    plt.title(name + "层次聚类调参折线图")
    plt.xlabel("t")
    plt.ylabel("轮廓系数")
    plt.savefig('HC调参折线图_' + name + '.png', dpi=300)
    plt.show()

    # 根据评价指标选择最佳的t值，这里以轮廓系数为主要依据
    best_t = 2 if name == "原始数据" else 2
    print(name + "最佳的t值为：", best_t)

    # 使用最佳的t值进行层次聚类算法（使用平均距离法）
    Z = linkage(data, method="average")
    labels = fcluster(Z, best_t, criterion="maxclust")

    # 计算轮廓系数
    silhouette_score = metrics.silhouette_score(data, labels)

    # 输出聚类结果的基本信息
    print(name + "层次聚类结果：")
    print("聚类个数：", len(np.unique(labels)))
    print("轮廓系数：", silhouette_score)

    # 绘制聚类树状图（dendrogram）
    plt.figure(figsize=(18, 10))
    dendrogram(Z)
    plt.title(name + "层次聚类树状图")
    plt.xlabel("样本索引")
    plt.ylabel("距离")
    plt.savefig('HC树状图_' + name + '.png', dpi=300)
    plt.show()
    # 绘制聚类结果的散点图（根据数据集的维度选择二维或三维）
    if name == "原始数据":
        # 绘制三维散点图
        fig = plt.figure(figsize=(10, 6)) # 创建一个图形对象
        ax = fig.add_subplot(111, projection='3d') # 创建一个三维子图对象
        ax.scatter(data["duration"], data["up_flow"], data["down_flow"], c=labels, cmap="rainbow") # 使用不同的颜色表示不同的聚类标签，并绘制三维散点图
        ax.set_title(name + "层次聚类结果散点图")
        ax.set_xlabel("duration")
        ax.set_ylabel("up_flow")
        ax.set_zlabel("down_flow")
        plt.savefig('HC散点图_原数据' + name + '.png', dpi=300)
        plt.show()
    else:
        # 绘制二维散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(data["pca1"], data["pca2"], c=labels, cmap="rainbow") # 使用不同的颜色表示不同的聚类标签，并绘制二维散点图
        plt.title(name + "层次聚类结果散点图")
        plt.xlabel("pca1")
        plt.ylabel("pca2")
        plt.savefig('HC散点图_pca' + name + '.png', dpi=300)
        plt.show()