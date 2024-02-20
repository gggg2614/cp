import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import datetime


def evaluate_model_with_subset(model, feature_subset, X_train, y_train, X_test, y_test):
    try:
        model.fit(X_train[list(feature_subset)], y_train)
        y_pred = model.predict(X_test[list(feature_subset)])
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2, feature_subset

    except Exception as e:
        print(f"Error in evaluating model with subset {feature_subset}: {e}")
        return float('inf'), -1, feature_subset
    
# 读取数据
filepath = "E:/LBE/T91/"
filename = "T91_data_noSarO.xlsx"

full_path = os.path.join(filepath, filename)

try:
    data = pd.read_excel(full_path)
    print(f"Data read successfully from {full_path}")
    
except Exception as e:
    print(f"Error reading file {full_path}: {e}")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 初始化标准化器
X_scaler = StandardScaler()
y_scaler = StandardScaler()

# 对X进行标准化
uni_X = X_scaler.fit_transform(X)
# 将标准化后的数组转换回DataFrame
uni_X_df = pd.DataFrame(uni_X, columns=X.columns)

# 对y进行标准化
uni_y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
# 将标准化后的数组转换回Series
uni_y_series = pd.Series(uni_y)

# 拆分训练集和测试集，保持为DataFrame和Series
X_train, X_test, y_train, y_test = train_test_split(uni_X_df, uni_y_series, test_size=0.3, random_state=42)

# 定义不同的模型
models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }


# 初始化保存结果的DataFrame
results_df = pd.DataFrame(columns=['Model', 'Feature Set', 'MSE', 'R2'])
csv_file_path = f'{filename}_R2_top_10_features_overall_per_model.csv'  # CSV文件路径

print('================================================')
print(f'filename = {filename}')
print('================================================')

# 对每种模型独立进行特征选择
for model_name, model in models.items():
    print(f"Evaluating model: {model_name}")

    model_results = []

    for subset_size in range(1, len(data.columns)):
        print(f"Processing feature subset size: {subset_size}")
        all_subsets = list(combinations(data.columns[:-1], subset_size))  # 除去目标列

        with tqdm(total=len(all_subsets), desc=f"Processing {model_name}, Subset size: {subset_size}") as pbar:
            results = Parallel(n_jobs=32)(
                delayed(evaluate_model_with_subset)(model, subset, X_train, y_train, X_test, y_test)
                for subset in all_subsets
                )
            
            model_results.extend(results)
            pbar.update(len(all_subsets))
            
        print(f"Completed {model_name} feature size {subset_size} at {datetime.datetime.now()}")
        # 对所有结果排序并获取前10个
    top_10_results = sorted(model_results, key=lambda x: x[0])[:10]

    # 将前10名结果添加到DataFrame中
    for mse, r2, subset in top_10_results:
        new_row = pd.DataFrame(
            {
            'Model': [model_name],
            'Feature Set': [', '.join(subset)],
            'MSE': [mse],
            'R2': [r2]
            }
        )
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # 每处理完一个模型，保存当前结果
    results_df.to_csv(csv_file_path, index=False)

    print(f"Completed processing all subset sizes for {model_name} at {datetime.datetime.now()}")

print(f"All models evaluated and results saved to {csv_file_path}")

print('all_finish')