import pandas as pd

for i in range(1, 22):
    # 逐个处理每个文件并将结果保存到临时文件
    df = pd.read_csv(f'day{i:02d}.txt', header=None, sep=',')
    df.columns = ['user_id', 'app_id', 'app_type', 'start_day', 'start_time', 'end_day', 'end_time', 'duration', 'up_flow', 'down_flow']

    # 将处理后的数据保存到临时文件
    df.to_csv(f'df_temp{i:02d}.csv', index=False)

    # 释放内存，删除临时数据框
    del df

    # 打印当前进度
    print(f"Processed file: day{i:02d}.txt")
