import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

plt.rcParams["font.family"] = "STSong"
pd.options.display.float_format = '{:.2f}'.format

# 导入minisom库
from minisom import MiniSom

# 读取特征数据框
df_cluster_features = pd.read_csv("df_cluster_features_guiyi.csv")
df_pca = pd.read_csv("df_pca.csv")

'''SOM clustering'''
# 定义两种数据集，一种是原始数据，一种是PCA降维后的数据
datasets = {"原始数据": df_cluster_features[["duration", "up_flow", "down_flow"]], "PCA降维后的数据": df_pca}

# 使用循环来对两种数据集分别进行SOM聚类
for name, data in datasets.items():
    # 创建一个SOM对象，指定竞争层的大小和迭代次数
    som = MiniSom(x=2, y=2, input_len=data.shape[1], sigma=0.5, learning_rate=0.5)
    # 初始化权重矩阵
    som.random_weights_init(data.values)
    # 训练SOM网络
    som.train_random(data.values, 100)
    # 得到每个样本对应的优胜节点
    labels = np.array([som.winner(x) for x in data.values])
    # 计算轮廓系数
    labels = np.dot(labels, [2, 1])
    silhouette_score = metrics.silhouette_score(data, labels)

    # 输出聚类结果的基本信息
    print(name + "SOM聚类结果：")
    print("聚类个数：", len(np.unique(labels)))
    print("轮廓系数：", silhouette_score)

    # 绘制距离图，显示不同节点之间的距离
    plt.figure(figsize=(10, 6))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.title(name + "SOM距离图")
    plt.colorbar()
    plt.savefig('SOM距离图_' + name + '.png', dpi=300)
    plt.show()

    # 绘制聚类结果的散点图（根据数据集的维度选择二维或三维）
    if name == "原始数据":
        # 绘制三维散点图
        fig = plt.figure(figsize=(10, 6))  # 创建一个图形对象
        ax = fig.add_subplot(111, projection='3d')  # 创建一个三维子图对象
        ax.scatter(data["duration"], data["up_flow"], data["down_flow"], c=labels, cmap="rainbow")  # 使用不同的颜色表示不同的聚类标签，并绘制三维散点图
        ax.set_title(name + "SOM聚类结果散点图")
        ax.set_xlabel("duration")
        ax.set_ylabel("up_flow")
        ax.set_zlabel("down_flow")
        plt.savefig('SOM散点图_原数据' + name + '.png', dpi=300)
        plt.show()
    else:
        # 绘制二维散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(data["pca1"], data["pca2"], c=labels,
                    cmap="rainbow")  # 使用不同的颜色表示不同的聚类标签，并绘制二维散点图
        plt.title(name + "SOM聚类结果散点图")
        plt.xlabel("pca1")
        plt.ylabel("pca2")
        plt.savefig('SOM散点图_pca' + name + '.png', dpi=300)
        plt.show()
