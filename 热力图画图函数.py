# 绘制聚类结果的热力图（使用原始数据）
# 定义热力图的函数
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def heatmap_plot(df, labels, n_clusters, title):
    # 获取数据特征的名称
    features = list(df.columns)

    # 创建一个空的数据框，存储每个聚类的平均值
    df_heatmap = pd.DataFrame(columns=features)

    # 使用循环来计算每个聚类的平均值，并添加到数据框中
    for i in range(n_clusters):
        # 获取第i个聚类的样本索引和平均值
        cluster_index = np.where(labels == i)[0]
        cluster_mean = df.iloc[cluster_index].mean()

        # 将平均值添加到数据框中
        df_heatmap.loc[len(df_heatmap)] = cluster_mean

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("特征")
    plt.ylabel("聚类")

    # 保存图像并显示
    plt.savefig(title + '.png', dpi=300)
    plt.show()