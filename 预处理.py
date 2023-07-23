import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="STSong"
pd.options.display.float_format = '{:.2f}'.format

# 定义一个空的数据框，用于存放所有天数的数据
df_1_11 = pd.DataFrame()
df_12_21 = pd.DataFrame()

# 读取单机app.csv文件，提取其中的单机app的app_id
df_single_app = pd.read_csv("所有app中的单机app.csv")
single_app_ids = df_single_app["app_id"].unique()
def time_to_period(time):
    hour = int(time[:2])
    if hour < 6:
        return '凌晨'
    elif hour < 12:
        return '上午'
    elif hour < 18:
        return '下午'
    else:
        return '晚上'

# 处理1~11的数据
for i in range(1, 12):
    df_temp = pd.read_csv(f'df_temp{i:02d}.csv')
    df_1_11 = pd.concat([df_1_11, df_temp], ignore_index=True)
    print(i,'已并入')

    # 删除临时文件
    del df_temp

# 删除重复的行
df_1_11.drop_duplicates(inplace=True)


# 查看是否存在缺失值，并将结果存放在一个新的数据框中
df_missing_1_11 = pd.DataFrame(df_1_11.isnull().sum(), columns=['missing_count'])
df_missing_1_11.to_csv('df_missing_1_11.csv', index=False)
missing_rows_1_11 = df_1_11[df_1_11.isnull().any(axis=1)]
missing_rows_1_11.to_csv('missing_rows_1_11.csv', index=False)
# 使用上一行的元素来替代缺失值
df_1_11.fillna(method='ffill', inplace=True)

# 释放内存，删除不再需要的 DataFrame 对象
del df_missing_1_11

#异常值处理
df_1_11 = df_1_11[(df_1_11['duration'] >= 10) & ((df_1_11['up_flow'] > 0) | (df_1_11['down_flow'] > 0) | (df_1_11['app_id'].isin(single_app_ids)))]

# 对'app_type'进行转换
df_1_11['app_type'] = df_1_11['app_type'].map({'sys': 0, 'usr': 1})
'''时间段的量化'''
# 将'start_time'和'end_time'转换为字符串类型
df_1_11['start_time'] = df_1_11['start_time'].astype(str)
df_1_11['end_time'] = df_1_11['end_time'].astype(str)
# 对'start_time'和'end_time'进行时间段转换
df_1_11['start_period'] = df_1_11['start_time'].apply(time_to_period)
df_1_11['end_period'] = df_1_11['end_time'].apply(time_to_period)
# 将时间段映射为数字
df_1_11['start_period'] = df_1_11['start_period'].map({'凌晨': 0, '上午': 1, '下午': 2, '晚上': 3})
df_1_11['end_period'] = df_1_11['end_period'].map({'凌晨': 0, '上午': 1, '下午': 2, '晚上': 3})
df_1_11.drop(['start_time', 'end_time'], axis=1, inplace=True)

# 保存为csv文件
df_1_11.to_csv('df_1_11.csv', index=False)

# 基于原始数据框，创建一个新的数据框，用于进行量化和聚类分析
# 删除不需要的列
df_1_11.drop(['start_day', 'end_day', 'end_period'], axis=1, inplace=True)
# 统计每个用户在df_1_11中出现的次数
user_counts_1_11 = df_1_11['user_id'].value_counts()

# 对'user_id'和'app_id'和'app_type','start_period'进行分组并求和
df_new_1_11 = df_1_11.groupby(['user_id', 'app_id','app_type','start_period']).sum()

# 重置索引
df_new_1_11.reset_index(inplace=True)

# 将'user_id'设置为行名
df_new_1_11.set_index('user_id', inplace=True)

# 将df_new保存为csv文件
df_new_1_11.to_csv('df_new_1_11.csv', index=True)

# 释放内存，删除不再需要的 DataFrame 对象
del df_1_11

# 对每个指标进行统计性计算，并将结果存放在一个新的数据框中
df_stats_1_11 = pd.DataFrame()
df_stats_1_11['mean'] = df_new_1_11.mean()
df_stats_1_11['median'] = df_new_1_11.median()
df_stats_1_11['max'] = df_new_1_11.max()
df_stats_1_11['min'] = df_new_1_11.min()
df_stats_1_11['var'] = df_new_1_11.var()
df_stats_1_11['std'] = df_new_1_11.std()
df_stats_1_11['q1'] = df_new_1_11.quantile(0.25)
df_stats_1_11['q3'] = df_new_1_11.quantile(0.75)
print(df_stats_1_11)

# 将df_stats保存为csv文件
df_stats_1_11.to_csv('df_stats_1_11.csv', index=False)
del df_stats_1_11

# 使用箱线图来检测异常值，并将所有指标的箱线图画在一个可视化图上
plt.figure(figsize=(12,8))
plt.boxplot([df_new_1_11['duration'], df_new_1_11['up_flow'], df_new_1_11['down_flow']],
            labels=['Duration', 'Up Flow', 'Down Flow'])
plt.title('箱线图')
plt.savefig('箱线图_1_11.png',dpi=300)
plt.show()

del df_new_1_11

# 处理12~21的数据
for i in range(12, 22):
    df_temp = pd.read_csv(f'df_temp{i:02d}.csv')
    df_12_21 = pd.concat([df_12_21, df_temp], ignore_index=True)
    print(i,'已并入')

    # 删除临时文件
    del df_temp

# 删除重复的行
df_12_21.drop_duplicates(inplace=True)

# 查看是否存在缺失值，并将结果存放在一个新的数据框中
df_missing_12_21 = pd.DataFrame(df_12_21.isnull().sum(), columns=['missing_count'])
df_missing_12_21.to_csv('df_missing_12_21.csv', index=False)
missing_rows_12_21 = df_12_21[df_12_21.isnull().any(axis=1)]
missing_rows_12_21.to_csv('missing_rows_12_21.csv', index=False)
# 使用上一行的元素来替代缺失值
df_12_21.fillna(method='ffill', inplace=True)

# 释放内存，删除不再需要的 DataFrame 对象
del df_missing_12_21

#异常值处理
df_12_21 = df_12_21[(df_12_21['duration'] >= 10) & ((df_12_21['up_flow'] > 0) | (df_12_21['down_flow'] > 0) | (df_12_21['app_id'].isin(single_app_ids)))]

# 对'app_type'进行转换
df_12_21['app_type'] = df_12_21['app_type'].map({'sys': 0, 'usr': 1})

'''时间段的量化'''
# 将'start_time'和'end_time'转换为字符串类型
df_12_21['start_time'] = df_12_21['start_time'].astype(str)
df_12_21['end_time'] = df_12_21['end_time'].astype(str)
# 对'start_time'和'end_time'进行时间段转换
df_12_21['start_period'] = df_12_21['start_time'].apply(time_to_period)
df_12_21['end_period'] = df_12_21['end_time'].apply(time_to_period)
# 将时间段转换为数字
df_12_21['start_period'] = df_12_21['start_period'].map({'凌晨': 0, '上午': 1, '下午': 2, '晚上': 3})
df_12_21['end_period'] = df_12_21['end_period'].map({'凌晨': 0, '上午': 1, '下午': 2, '晚上': 3})
df_12_21.drop(['start_time', 'end_time'], axis=1, inplace=True)


# 保存为csv文件
df_12_21.to_csv('df_12_21.csv', index=False)

# 基于原始数据框，创建一个新的数据框，用于进行量化和聚类分析
# 删除不需要的列
df_12_21.drop(['start_day',  'end_day', 'end_period'], axis=1, inplace=True)
# 统计每个用户在df_12_21中出现的次数
user_counts_12_21 = df_12_21['user_id'].value_counts()


# 对'user_id'和'app_id'和'app_type',start_period进行分组并求和
df_new_12_21 = df_12_21.groupby(['user_id', 'app_id', 'app_type','start_period']).sum()

# 重置索引
df_new_12_21.reset_index(inplace=True)

# 将'user_id'设置为行名
df_new_12_21.set_index('user_id', inplace=True)

# 将df_new保存为csv文件
df_new_12_21.to_csv('df_new_12_21.csv', index=True)

# 释放内存，删除不再需要的 DataFrame 对象
del df_12_21

# 对每个指标进行统计性计算，并将结果存放在一个新的数据框中
df_stats_12_21 = pd.DataFrame()
df_stats_12_21['mean'] = df_new_12_21.mean()
df_stats_12_21['median'] = df_new_12_21.median()
df_stats_12_21['max'] = df_new_12_21.max()
df_stats_12_21['min'] = df_new_12_21.min()
df_stats_12_21['var'] = df_new_12_21.var()
df_stats_12_21['std'] = df_new_12_21.std()
df_stats_12_21['q1'] = df_new_12_21.quantile(0.25)
df_stats_12_21['q3'] = df_new_12_21.quantile(0.75)
print(df_stats_12_21)

# 将df_stats保存为csv文件
df_stats_12_21.to_csv('df_stats_12_21.csv', index=False)
del df_stats_12_21

# 使用箱线图来检测异常值，并将所有指标的箱线图画在一个可视化图上
plt.figure(figsize=(12,8))
plt.boxplot([df_new_12_21['duration'], df_new_12_21['up_flow'], df_new_12_21['down_flow']],
            labels=['Duration', 'Up Flow', 'Down Flow'])
plt.title('箱线图')
plt.savefig('箱线图_12_21.png',dpi=300)
plt.show()
del df_new_12_21

#合并df_new_1_11和df_new_12_21
df_new_1_11 = pd.read_csv('df_new_1_11.csv')
df_new_12_21 = pd.read_csv('df_new_12_21.csv')
merged_df = pd.merge(df_new_1_11, df_new_12_21, on=['user_id', 'app_id','app_type','start_period'], suffixes=('_1_11', '_12_21'),how='left')
merged_df['duration'] = merged_df['duration_1_11'] + merged_df['duration_12_21'].fillna(0)
merged_df['up_flow'] = merged_df['up_flow_1_11'] + merged_df['up_flow_12_21'].fillna(0)
merged_df['down_flow'] = merged_df['down_flow_1_11'] + merged_df['down_flow_12_21'].fillna(0)
del df_new_1_11,df_new_12_21

# 创建新的数据框df_new
df_new = merged_df[['user_id', 'app_id', 'app_type','start_period','duration', 'up_flow', 'down_flow',]]
del merged_df
pd.options.display.float_format = '{:.2f}'.format
print(df_new)
df_new.to_csv('df_new.csv', index=False)

# 合并df_1_11和df_12_21的用户出现次数，对于共同出现的用户次数相加，对于单独出现的用户保留各自的次数
user_counts = user_counts_1_11.combine_first(user_counts_12_21)

# 初始化空的数据框df_cluster_features
df_user_counts = pd.DataFrame()
# 创建新的数据框df_cluster_features
df_user_counts['user_id'] = user_counts.index
df_user_counts['usage_count'] = user_counts.values

# 读取辅助表格app_class.csv
app_class = pd.read_csv('app_class.csv',header=None)

# 提取常用所属的20类app_id
common_app_ids = app_class.iloc[:, 0].unique()

# 根据常用app_id进行筛选
df_common = df_new[df_new['app_id'].isin(common_app_ids)]
df_common = df_common.reset_index(drop=True)
print(df_common)

# 按用户分组并求和
df_sum = df_common.groupby('user_id').sum()

# 创建新的数据框df_features
df_cluster_features = df_sum.reset_index()

# 仅保留用户ID、duration、up_flow和down_flow列
df_cluster_features = df_cluster_features[['user_id','duration', 'up_flow', 'down_flow']]

# 在df_cluster_features中新增'usage_count'一列，并填充对应的值
df_cluster_features['usage_count'] = df_cluster_features['user_id'].map(df_user_counts.set_index('user_id')['usage_count'])

# 计算'avg_duration'、'avg_up_flow'、'avg_down_flow'列的值
df_cluster_features['avg_duration'] = df_cluster_features['duration'] / df_cluster_features['usage_count']
df_cluster_features['avg_up_flow'] = df_cluster_features['up_flow'] / df_cluster_features['usage_count']
df_cluster_features['avg_down_flow'] = df_cluster_features['down_flow'] / df_cluster_features['usage_count']

print(df_cluster_features)
df_cluster_features.to_csv('df_cluster_features.csv', index=False)