import pandas as pd

# 读取数据框
df_cluster_features = pd.read_csv("df_cluster_features.csv")
df_cluster_features = df_cluster_features[['user_id', 'usage_count']]
df_kmeans_cluster_result = pd.read_csv('df_kmeans_cluster_result.csv')
df_new = pd.read_csv('df_new.csv')
app_class = pd.read_csv('app_class.csv', header=None)

# 添加列名
app_class.columns = ['app_id', 'app_type']

# 初始化统计数据框
df_stat = df_kmeans_cluster_result.copy()

# 初始化a-t列
df_stat[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']] = 0

# 初始化凌晨、上午、中午、晚上列
df_stat[['凌晨', '上午', '中午', '晚上']] = 0

# 初始化总流量列和使用次数列
df_stat['总流量'] = 0
df_stat['使用次数'] = 0

# 遍历df_kmeans_cluster_result的每一行
for index, row in df_stat.iterrows():
    user_id = row['user_id']

    # 统计a-t列
    user_data = df_new[df_new['user_id'] == user_id]
    for app_id, duration in zip(user_data['app_id'], user_data['duration']):
        if app_id in app_class['app_id'].values:
            app_type = app_class[app_class['app_id'] == app_id]['app_type'].values[0]
            df_stat.at[index, app_type] += duration

    # 统计凌晨、上午、中午、晚上列
    for period, duration in zip(user_data['start_period'], user_data['duration']):
        df_stat.at[index, ['凌晨', '上午', '中午', '晚上'][period]] += duration

    # 统计总流量
    if user_id in df_new['user_id'].values:
        user_flow = df_new[df_new['user_id'] == user_id]
        total_flow = user_flow['up_flow'].sum() + user_flow['down_flow'].sum()
        df_stat.at[index, '总流量'] = total_flow

    # 统计使用次数
    if user_id in df_cluster_features['user_id'].values:
        usage_count = df_cluster_features[df_cluster_features['user_id'] == user_id]['usage_count'].values[0]
        df_stat.at[index, '使用次数'] = usage_count

# 输出统计结果
print(df_stat)

# 保存结果到CSV文件
df_stat.to_csv('聚类后画像.csv', index=False)

