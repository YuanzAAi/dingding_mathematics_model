import pandas as pd

# 读取数据
df_1_11 = pd.read_csv("df_1_11.csv")
df_12_21 = pd.read_csv("df_12_21.csv")
app_class = pd.read_csv('app_class.csv', header=None)

# 删除不需要的列
df_1_11.drop(['start_day', 'end_day', 'start_period', 'end_period', 'user_id', 'app_type'], axis=1, inplace=True)
df_12_21.drop(['start_day', 'end_day', 'start_period', 'end_period', 'user_id', 'app_type'], axis=1, inplace=True)

# 合并df_1_11和df_12_21，并根据app_id进行分组并求和
df_merged = pd.concat([df_1_11, df_12_21]).groupby('app_id').sum()

# 输出合并后的结果
print(df_merged)

# 统计down_flow和up_flow同时为0的行数
num_rows_with_zero_flow = len(df_merged[(df_merged['down_flow'] == 0) & (df_merged['up_flow'] == 0)])

# 输出down_flow和up_flow同时为0的app_id
app_ids_with_zero_flow = df_merged[(df_merged['down_flow'] == 0) & (df_merged['up_flow'] == 0)].index.tolist()

# 将down_flow和up_flow同时为0的app_id保存成数据框并以'单机app.csv'的文件名保存
df_app_ids_with_zero_flow = pd.DataFrame(app_ids_with_zero_flow, columns=['app_id'])
df_app_ids_with_zero_flow.to_csv('所有app中的单机app.csv', index=False)

# 根据常用app_id进行筛选
common_app_ids = app_class.iloc[:, 0].unique()
df_common = df_merged[df_merged.index.isin(common_app_ids)]

# 统计常见20类app中down_flow和up_flow同时为0的行数
num_common_rows_with_zero_flow = len(df_common[(df_common['down_flow'] == 0) & (df_common['up_flow'] == 0)])

# 输出常见20类app中down_flow和up_flow同时为0的app_id
common_app_ids_with_zero_flow = df_common[(df_common['down_flow'] == 0) & (df_common['up_flow'] == 0)].index.tolist()

# 将常见20类app中down_flow和up_flow同时为0的app_id保存成数据框并以'常见20类单机app.csv'的文件名保存
df_common_app_ids_with_zero_flow = pd.DataFrame(common_app_ids_with_zero_flow, columns=['app_id'])
df_common_app_ids_with_zero_flow.to_csv('常见20类app中的单机app.csv', index=False)

# 打印结果
print("所有app中down_flow和up_flow同时为0的行数:", num_rows_with_zero_flow)
print("常见20类app中down_flow和up_flow同时为0的行数:", num_common_rows_with_zero_flow)
