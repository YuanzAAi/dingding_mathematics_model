import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
plt.rcParams["font.family"] = "STSong"
pd.options.display.float_format = '{:.2f}'.format

from sklearn.cluster import Birch

# 读取特征数据框
df_cluster_features = pd.read_csv("df_cluster_features_guiyi.csv")
df_pca = pd.read_csv("df_pca.csv")

'''BIRCH clustering'''
# 定义两种数据集，一种是原始数据，一种是PCA降维后的数据
datasets = {"原始数据": df_cluster_features[["duration", "up_flow", "down_flow"]], "PCA降维后的数据": df_pca}

# 使用循环来对两种数据集分别进行BIRCH聚类
for name, data in datasets.items():
    '''调参'''
    # 定义threshold值的参数范围
    threshold_range = np.arange(0.01, 0.11, 0.01) if name == "原始数据" else np.arange(0.1, 0.21, 0.01)

    # 创建一个空的数据框，存储评价指标
    df_birch_evaluation = pd.DataFrame(columns=["threshold", "silhouette_score"])

    # 使用循环来尝试不同的threshold值，并计算轮廓系数
    for threshold in threshold_range:
        try:
            # 创建一个Birch对象，指定阈值、分支因子和聚类个数
            birch = Birch(threshold=threshold, branching_factor=50, n_clusters=4)
            # 对数据进行聚类，并得到每个样本的聚类标签
            labels = birch.fit_predict(data)

            # 计算轮廓系数
            silhouette_score = metrics.silhouette_score(data, labels)
            # 将结果添加到数据框中
            df_birch_evaluation.loc[len(df_birch_evaluation)] = [threshold, silhouette_score]
        except ValueError:
            silhouette_score = 0

    # 输出数据框
    print(name + "BIRCH聚类调参结果：")
    print(df_birch_evaluation)
    df_birch_evaluation.to_csv('df_birch_evaluation_' + name + '.csv', index=False)

    # 对评价指标进行可视化，使用折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df_birch_evaluation["threshold"], df_birch_evaluation["silhouette_score"])
    plt.title(name + "BIRCH聚类调参折线图")
    plt.xlabel("threshold")
    plt.ylabel("轮廓系数")
    plt.savefig('BIRCH调参折线图_' + name + '.png', dpi=300)
    plt.show()

    # 根据评价指标选择最佳的threshold值，这里以轮廓系数为主要依据
    # 从折线图中可以看出，当threshold为0.05时，轮廓系数最高，因此选择0.05作为最佳的threshold值
    best_threshold = 0.09 if name == "原始数据" else 0.16
    print(name + "最佳的threshold值为：", best_threshold)
    '''调参'''

    # 使用最佳的threshold值创建一个Birch对象，指定阈值、分支因子和聚类个数
    birch = Birch(threshold=best_threshold, branching_factor=50, n_clusters=4)
    # 对数据进行聚类，并得到每个样本的聚类标签
    labels = birch.fit_predict(data)
    # 得到每个子簇的质心
    centers = birch.subcluster_centers_
    # 得到每个子簇的全局聚类标签
    sublabels = birch.subcluster_labels_

    # 计算轮廓系数
    silhouette_score = metrics.silhouette_score(data, labels)

    # 输出聚类结果的基本信息
    print(name + "BIRCH聚类结果：")
    print("聚类个数：", len(np.unique(labels)))
    print("轮廓系数：", silhouette_score)

    # 绘制聚类结果的散点图（根据数据集的维度选择二维或三维）
    if name == "原始数据":
        # 绘制三维散点图
        fig = plt.figure(figsize=(10, 6))  # 创建一个图形对象
        ax = fig.add_subplot(111, projection='3d')  # 创建一个三维子图对象
        ax.scatter(data["duration"], data["up_flow"], data["down_flow"], c=labels,
                   cmap="rainbow")  # 使用不同的颜色表示不同的聚类标签，并绘制三维散点图
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=sublabels, marker="*", s=200,
                   edgecolors="k")  # 使用不同的颜色和形状表示不同的子簇质心，并绘制三维散点图
        ax.set_title(name + "BIRCH聚类结果散点图")
        ax.set_xlabel("duration")
        ax.set_ylabel("up_flow")
        ax.set_zlabel("down_flow")
        plt.savefig('BIRCH散点图_原数据' + name + '.png', dpi=300)
        plt.show()
    else:
        # 绘制二维散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(data["pca1"], data["pca2"], c=labels, cmap="rainbow")  # 使用不同的颜色表示不同的聚类标签，并绘制二维散点图
        plt.scatter(centers[:, 0], centers[:, 1], c=sublabels, marker="*", s=200,
                    edgecolors="k")  # 使用不同的颜色和形状表示不同的子簇质心，并绘制二维散点图
        plt.title(name + "BIRCH聚类结果散点图")
        plt.xlabel("pca1")
        plt.ylabel("pca2")
        plt.savefig('BIRCH散点图_pca' + name + '.png', dpi=300)
        plt.show()
