# 导入所需的库
import pandas as pd

# 读取数据文件
app_class = pd.read_csv('app_class.csv', header=None, names=['app_id', 'class'])
df_1_11 = pd.read_csv('df_1_11.csv')
df_12_21 = pd.read_csv('df_12_21.csv')

# 对齐用户，去掉只在一个数据集中出现的用户
users = set(df_1_11['user_id']).intersection(set(df_12_21['user_id']))
df_1_11 = df_1_11[df_1_11['user_id'].isin(users)]
df_12_21 = df_12_21[df_12_21['user_id'].isin(users)]


# 定义一个函数，用于对每个数据集进行特征提取
def feature_extraction(df):
    # 按用户分组，计算各种统计量
    features = df.groupby('user_id').agg({
        'app_id': ['count'],  # 使用次数
        'duration': ['sum', 'mean', 'max', 'std', 'skew'],  # 使用时长
        'up_flow': ['sum', 'mean', 'max', 'std', 'skew'],  # 上行流量
        'down_flow': ['sum', 'mean', 'max', 'std', 'skew'],  # 下行流量
        # 使用start_period和end_period的众数表示使用时间段
        'start_period': lambda x: x.mode()[0],  # 使用起始时间段的众数
        'end_period': lambda x: x.mode()[0],  # 使用结束时间段的众数
        'start_day': lambda x: x.nunique() / x.max(),  # 使用频率（使用天数占比）
        'end_day': lambda x: x.diff().mean(),  # 使用间隔（平均相邻两次使用的天数差）
    })
    # 重命名列名，去掉多层索引
    features.columns = ['_'.join(col) for col in features.columns.values]
    features.reset_index(inplace=True)
    # 计算有效日均时长，即每天至少使用一次的情况下，每天的平均使用时长
    daily_duration = df.groupby(['user_id', 'start_day'])['duration'].agg(['count', 'sum'])
    daily_duration['avg'] = daily_duration['sum'] / daily_duration['count']
    daily_duration.reset_index(inplace=True)
    effective_daily_duration = daily_duration.groupby('user_id')['avg'].mean()
    # 将有效日均时长添加到特征中
    features = features.merge(effective_daily_duration, on='user_id', how='left')
    features.rename(columns={'avg': 'effective_daily_duration'}, inplace=True)

    # 增加最大次数、最大时长、最大流量，日均次数、日均时长、日均流量这些特征的计算
    # 按用户和天分组，计算每天的使用次数、使用时长、上行流量、下行流量
    daily_stats = df.groupby(['user_id', 'start_day']).agg({
        'app_id': ['count'],  # 每天的使用次数
        'duration': ['sum'],  # 每天的使用时长
        'up_flow': ['sum'],  # 每天的上行流量
        'down_flow': ['sum'],  # 每天的下行流量
    })
    # 重命名列名，去掉多层索引
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns.values]
    daily_stats.reset_index(inplace=True)
    # 按用户分组，计算每天的最大值和平均值
    daily_max_mean = daily_stats.groupby('user_id').agg({
        'app_id_count': ['max', 'mean'],  # 最大次数和日均次数
        'duration_sum': ['max', 'mean'],  # 最大时长和日均时长
        'up_flow_sum': ['max', 'mean'],  # 最大上行流量和日均上行流量
        'down_flow_sum': ['max', 'mean'],  # 最大下行流量和日均下行流量
    })
    # 重命名列名，去掉多层索引
    daily_max_mean.columns = ['_'.join(col) for col in daily_max_mean.columns.values]
    daily_max_mean.reset_index(inplace=True)
    # 将每天的最大值和平均值添加到特征中
    features = features.merge(daily_max_mean, on='user_id', how='left')
    features.rename(columns={'app_id_count': 'total_count', 'app_id_count_max': 'max_day_count','app_id_count_mean':'mean_day_count','duration_sum_max':'max_day_duration',
                             'duration_sum_mean':'mean_day_duration','up_flow_sum_max':'max_day_up_flow',
                             'up_flow_sum_mean':'mean_day_up_flow','down_flow_sum_max':'max_day_down_flow',
                             'down_flow_sum_mean':'mean_day_down_flow','start_day_<lambda>':'freq','end_day_<lambda>':'gap'}, inplace=True)
    # 返回特征数据框
    return features


# 对df_1_11和df_12_21分别进行特征提取
features_1_11 = feature_extraction(df_1_11)
features_12_21 = feature_extraction(df_12_21)


# 定义一个函数，用于对每个数据集进行标签生成
def label_generation(df, app_class):
    # 合并数据框，得到每个APP的类别
    df = df.merge(app_class, on='app_id', how='left')
    # 创建一个空的数据框，用于存储标签
    labels = pd.DataFrame()
    # 获取所有用户的ID，并添加到标签数据框中
    labels['user_id'] = df['user_id'].unique()
    # 先将use_a列全部填充为0
    labels['use_a'] = 0
    # 筛选出只包含a类APP的数据
    a_class_data = df[df['class'] == 'a']
    # 计算使用a类APP的有效日均时长（与之前计算整体有效日均时长的方法相同）
    daily_duration_a = a_class_data.groupby(['user_id', 'start_day'])['duration'].agg(['count', 'sum'])
    daily_duration_a['avg'] = daily_duration_a['sum'] / daily_duration_a['count']
    daily_duration_a.reset_index(inplace=True)
    effective_daily_duration_a = daily_duration_a.groupby('user_id')['avg'].mean()
    # 将计算结果添加到标签数据框中
    labels['effective_daily_duration'] = labels['user_id'].map(effective_daily_duration_a)
    # 将使用过a类APP的用户的use_a列设置为1
    labels.loc[labels['effective_daily_duration'].notnull(), 'use_a'] = 1
    # 填充缺失值为0，否则会影响后续的模型训练
    labels['effective_daily_duration'] = labels['effective_daily_duration'].fillna(0)
    # 返回标签数据框
    return labels

# 对df_1_11和df_12_21分别进行标签生成
labels_1_11 = label_generation(df_1_11, app_class)
labels_12_21 = label_generation(df_12_21, app_class)

# 将特征和标签合并，得到训练集和测试集
train = features_1_11.merge(labels_1_11, on='user_id', how='left')
test = features_12_21.merge(labels_12_21, on='user_id', how='left')

# 填充缺失值为0，确保训练集和测试集中没有缺失值
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# 保存训练集和测试集为csv文件
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
