import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import MeanShift,estimate_bandwidth # 导入均值漂移聚类算法
plt.rcParams["font.family"]="STSong"
pd.options.display.float_format = '{:.2f}'.format
from 热力图画图函数 import heatmap_plot


# 读取特征数据框
df_cluster_features = pd.read_csv("df_cluster_features_guiyi.csv")
df_pca = pd.read_csv("df_pca.csv")

'''mean shift'''
# 定义两种数据集，一种是原始数据，一种是PCA降维后的数据
datasets = {"原始数据": df_cluster_features[["duration", "up_flow", "down_flow"]], "PCA降维后的数据": df_pca}

# 使用循环来对两种数据集分别进行均值漂移聚类
for name, data in datasets.items():
    # 使用最佳的bandwidth值进行均值漂移聚类算法
    best_bandwidth = estimate_bandwidth(data, quantile=0.3, random_state=0, n_jobs=-1)
    print(name,'的最佳bandwidth为',best_bandwidth)
    mean_shift = MeanShift(bandwidth=best_bandwidth, n_jobs=-1)  # 指定bandwidth参数
    mean_shift.fit(data)
    labels = mean_shift.labels_
    n_clusters = len(set(labels))  # 计算聚类的个数，排除噪声点

    # 如果聚类个数大于1，就计算轮廓系数，否则设为-1
    if n_clusters > 1:
        silhouette_score = metrics.silhouette_score(data, labels)  # 计算轮廓系数
    else:
        silhouette_score = -1

    # 输出聚类结果的基本信息
    print(name + "均值漂移聚类结果：")
    print("聚类个数：", n_clusters)
    print("轮廓系数：", silhouette_score)

    # 绘制聚类结果的散点图
    if name == "原始数据":
        # 绘制三维散点图
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data["duration"], data["up_flow"], data["down_flow"], c=labels, cmap="rainbow")
        ax.set_title(name + "均值漂移聚类结果三维散点图")
        ax.set_xlabel("duration")
        ax.set_ylabel("up_flow")
        ax.set_zlabel("down_flow")
        plt.savefig('均值漂移聚类结果三维散点图_' + name + '.png', dpi=300)
        plt.show()
    else:
        # 绘制二维散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(data["pca1"], data["pca2"], c=labels, cmap="rainbow")
        plt.title(name + "均值漂移聚类结果散点图")
        plt.xlabel("pca1")
        plt.ylabel("pca2")
        plt.savefig('均值漂移聚类结果散点图_pca_' + name + '.png', dpi=300)
        plt.show()

    # 调用热力图的函数，传入数据特征，聚类标签，聚类个数和标题
    heatmap_plot(data, labels, n_clusters, name + "均值漂移聚类结果热力图")
