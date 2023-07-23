# 导入需要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline


# 读取训练集和测试集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 复制一份不做归一化的train0和test0
train0 = train.copy()
test0 = test.copy()

# 对除了start_period_<lambda>和end_period_<lambda>以外的数据进行归一化处理
scaler = MinMaxScaler()
columns_to_scale = [col for col in train.columns if col not in ['start_period_<lambda>', 'end_period_<lambda>', 'user_id', 'use_a', 'effective_daily_duration_y']]
train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
test[columns_to_scale] = scaler.transform(test[columns_to_scale])

# 更改列名
test.rename(columns={'start_period_<lambda>': 'start_period', 'end_period_<lambda>': 'end_period'}, inplace=True)
train.rename(columns={'start_period_<lambda>': 'start_period', 'end_period_<lambda>': 'end_period'}, inplace=True)
test0.rename(columns={'start_period_<lambda>': 'start_period', 'end_period_<lambda>': 'end_period'}, inplace=True)
train0.rename(columns={'start_period_<lambda>': 'start_period', 'end_period_<lambda>': 'end_period'}, inplace=True)

# 根据选取的六个特征，分别获取归一化和未归一化的训练集和测试集的特征和标签
selected_features = ['down_flow_skew', 'freq', 'gap', 'down_flow_sum', 'max_day_count', 'mean_day_count']
X_train_scaled = train[selected_features]
y_train_scaled = train['effective_daily_duration_y']
X_test_scaled = test[selected_features]
y_test_scaled = test['effective_daily_duration_y']
X_train_unscaled = train0[selected_features]
y_train_unscaled = train0['effective_daily_duration_y']
X_test_unscaled = test0[selected_features]
y_test_unscaled = test0['effective_daily_duration_y']

# 定义一个函数，用于对不同的回归器进行参数调优，训练模型，并计算评价指标，返回一个包含回归器名字和评价指标的数据框
def evaluate_regressors(regressors, X_train, y_train, X_test, y_test):
    # 创建一个空列表，用于存储每个回归器的结果
    results = []
    # 遍历每个回归器及其参数网格
    for regressor_name, regressor, param_grid in regressors:
        # 使用网格搜索算法进行参数调优，使用5折交叉验证，使用NMSE作为评分标准，使用-1个核心并行计算
        grid_search = GridSearchCV(regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        # 在训练集上拟合网格搜索对象
        grid_search.fit(X_train, y_train)
        # 获取最优参数
        best_params = grid_search.best_params_
        # 打印最优参数
        print(f'{regressor_name} best parameters: {best_params}')
        # 使用最优参数的回归器在测试集上进行预测
        y_pred = grid_search.predict(X_test)
        # 计算NMSE、准确率和AUC值
        nmse = -mean_squared_error(y_test, y_pred)
        accuracy = accuracy_score(y_test > 0.5, y_pred > 0.5)
        auc = roc_auc_score(y_test > 0.5, y_pred > 0.5)
        # 将回归器名字和评价指标添加到结果列表中
        results.append([regressor_name, nmse, accuracy, auc])
    # 将结果列表转换为数据框，并设置列名
    df_results = pd.DataFrame(results, columns=['regressor', 'nmse', 'accuracy', 'auc'])
    # 返回数据框
    return df_results

# 定义一个函数，用于绘制多柱图，显示不同回归器的评价指标
def plot_evaluation_metrics(df_results):
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    # 设置柱状图的宽度
    bar_width = 0.15
    # 设置柱状图的位置
    x1 = np.arange(3)
    x2 = x1 + bar_width
    x3 = x2 + bar_width
    x4 = x3 + bar_width
    x5 = x4 + bar_width
    x6 = x5 + bar_width
    x7 = x6 + bar_width
    x8 = x7 + bar_width
    x9 = x8 + bar_width
    x10 = x9 + bar_width
    # 绘制线性回归的评价指标柱状图
    plt.bar(x1, df_results.iloc[0, 1:], width=bar_width, label='Linear Regression')
    # 绘制岭回归的评价指标柱状图
    plt.bar(x2, df_results.iloc[1, 1:], width=bar_width, label='Ridge Regression')
    # 绘制LASSO回归的评价指标柱状图
    plt.bar(x3, df_results.iloc[2, 1:], width=bar_width, label='LASSO Regression')
    # 绘制多项式回归的评价指标柱状图
    plt.bar(x4, df_results.iloc[3, 1:], width=bar_width, label='Polynomial Regression')
    # 绘制支持向量回归的评价指标柱状图
    plt.bar(x5, df_results.iloc[4, 1:], width=bar_width, label='Support Vector Regression')
    # 绘制朴素贝叶斯回归的评价指标柱状图
    plt.bar(x6, df_results.iloc[5, 1:], width=bar_width, label='Naive Bayes Regression')
    # 绘制KNN回归的评价指标柱状图
    plt.bar(x7, df_results.iloc[6, 1:], width=bar_width, label='KNN Regression')
    # 绘制多层感知机回归的评价指标柱状图
    plt.bar(x8, df_results.iloc[7, 1:], width=bar_width, label='Multilayer Perceptron Regression')
    # 绘制XGBoost回归的评价指标柱状图
    plt.bar(x9, df_results.iloc[8, 1:], width=bar_width, label='XGBoost Regression')
    # 绘制RF回归的评价指标柱状图
    plt.bar(x10, df_results.iloc[9, 1:], width=bar_width, label='RF Regression')

    # 设置x轴的刻度和标签，使用评价指标的名称
    plt.xticks(x5 + bar_width / 2, df_results.columns[1:])
    # 设置y轴的标签，使用分数作为单位
    plt.ylabel('得分')
    # 设置标题，使用回归器性能比较作为标题
    plt.title('回归评价指标')
    # 显示图例，使用回归器的名称作为图例，并设置位置在右上角
    plt.legend(loc='upper right')
    # 保存图片到本地，使用evaluation_metrics_comparison.png作为文件名，并设置分辨率为300dpi
    plt.savefig('回归评价指标.png', dpi=300)
    # 显示图片
    plt.show()


# 定义一个列表，存储不同的回归器及其参数网格，注意使用归一化和未归一化的数据分别创建回归器对象

regressors = [
    # 使用归一化数据创建线性回归器对象，并设置参数网格为空字典
    ('Linear Regression', LinearRegression(), {}),
    # 使用归一化数据创建岭回归器对象，并设置参数网格为正则化参数alpha在0.01到100之间取对数均匀分布的10个值
    ('Ridge Regression', Ridge(), {'alpha': np.logspace(-2, 2, 10)}),
    # 使用归一化数据创建LASSO回归器对象，并设置参数网格为正则化参数alpha在0.01到100之间取对数均匀分布的10个值
    ('LASSO Regression', Lasso(), {'alpha': np.logspace(-2, 2, 10)}),
    # 使用归一化数据创建多项式回归器对象，并设置参数网格为多项式次数degree在1到5之间取5个值，线性回归器linear_model为LinearRegression，Ridge或者Lasso
    ('Polynomial Regression', Pipeline([('poly', PolynomialFeatures()), ('linear_model', LinearRegression())]),
     {'poly__degree': range(1, 6), 'linear_model': [LinearRegression(), Ridge(), Lasso()]}),
    # 使用归一化数据创建支持向量回归器对象，并设置参数网格为正则化参数C在0.01到100之间取对数均匀分布的10个值，核函数kernel为rbf或者linear
    ('Support Vector Regression', SVR(), {'C': np.logspace(-2, 2, 10), 'kernel': ['rbf', 'linear']}),
    # 使用未归一化数据创建朴素贝叶斯回归器对象，并设置参数网格为空字典
    ('Naive Bayes Regression', GaussianNB(), {}),
    # 使用归一化数据创建KNN回归器对象，并设置参数网格为邻居个数n_neighbors在1到10之间取10个值，距离度量metric为minkowski或者euclidean
    ('KNN Regression', KNeighborsRegressor(), {'n_neighbors': range(1, 11), 'metric': ['minkowski', 'euclidean']}),
    # 使用归一化数据创建多层感知机回归器对象，并设置参数网格为隐藏层神经元个数hidden_layer_sizes在(10,)到(100,)之间取10个值，激活函数activation为relu或者tanh，最大迭代次数max_iter为1000或者使用早停法early_stopping为True
    ('Multilayer Perceptron Regression', MLPRegressor(),
     {'hidden_layer_sizes': [(i,) for i in range(10, 101, 10)], 'activation': ['relu', 'tanh'], 'max_iter': [1000],
      'early_stopping': [True]}),
    # 使用未归一化数据创建XGBoost回归器对象，并设置参数网格为树的个数n_estimators在100到1000之间取10个值，学习率learning_rate在0.01到1之间取对数均匀分布的10个值，最大深度max_depth在1到10之间取10个值
    ('XGBoost Regression', XGBRegressor(),
     {'n_estimators': range(100, 1001, 100), 'learning_rate': np.logspace(-2, 0, 10), 'max_depth': range(1, 11)}),
    # 使用未归一化数据创建RF回归器对象，并设置参数网格为树的个数n_estimators在100到1000之间取10个值，最大深度max_depth在1到10之间取10个值，最小分裂样本数min_samples_split在2到20之间取10个值
    ('RF Regression', RandomForestRegressor(),
     {'n_estimators': range(100, 1001, 100), 'max_depth': range(1, 11), 'min_samples_split': range(2, 21)})
]

# 使用归一化数据对线性回归、岭回归、LASSO回归、多项式回归和支持向量回归进行评价，返回一个数据框
df_results_scaled = evaluate_regressors(regressors[:5], X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

# 使用未归一化数据对朴素贝叶斯回归、XGBoost回归和RF回归进行评价，返回一个数据框
df_results_unscaled = evaluate_regressors(regressors[5:8], X_train_unscaled, y_train_unscaled, X_test_unscaled,
                                          y_test_unscaled)

# 使用归一化数据对KNN回归和多层感知机回归进行评价，返回一个数据框
df_results_scaled_2 = evaluate_regressors(regressors[8:], X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

# 将三个数据框合并为一个数据框
df_results = pd.concat([df_results_scaled, df_results_unscaled, df_results_scaled_2], ignore_index=True)
df_results.to_csv('各模型分类的回归结果.csv', index=False)

# 调用绘图函数，显示不同回归器的评价指标
plot_evaluation_metrics(df_results)

