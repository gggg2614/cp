import copy
import json
import math
import os
import re

from flask import Flask, jsonify, request, send_file, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MaxAbsScaler
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from joblib import Parallel, delayed, dump, load
from torch import nn, optim
from IPython.display import clear_output
import torch
import torch.utils.data as Data
from tqdm import tqdm
from gplearn.genetic import SymbolicTransformer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, supports_credentials=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置文件上传目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 设置临时文件目录
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'pth', 'joblib', 'txt'}


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b if b != 0 else float('inf')  # 避免除以零


def sqrt(a):
    return math.sqrt(a)


def log(a, base=10):
    return math.log(a, base)  # 默认为以10为底的对数


def neg(a):
    return -a


def inv(a):
    return 1 / a if a != 0 else float('inf')  # 避免除以0


def allowed_file(filename):
    filename = secure_filename(filename)
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload():
    # 检查是否存在上传文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 保存文件到临时文件夹
    filename = secure_filename(file.filename)
    temp_filepath = os.path.join(TEMP_FOLDER, filename)
    file.save(temp_filepath)

    # 读取文件内容
    if filename.endswith('.csv') or filename == 'csv':
        pass
    elif filename.endswith('.xlsx') or filename == 'xlsx':
        pass
    elif filename.endswith('.pth') or filename == 'pth':
        pass
    elif filename.endswith('.joblib') or filename == 'joblib':
        pass
    elif filename.endswith('.txt') or filename == 'txt':
        pass
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    # 返回临时文件路径
    return jsonify({'message': '文件上传成功', 'temp_file_path': temp_filepath})


@app.route('/file1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print(request)
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    n_cv = int(request.form['cv'])
    n_searches = int(request.form['n_searches'])
    population_size = request.form['population_size']
    population_size_min, population_size_max = map(int, population_size.split(','))

    generations = request.form['generations']
    generations_size_min, generations_size_max = map(int, generations.split(','))

    maxSamples = int(request.form['maxSamples'])

    pCrossover = request.form['pCrossover']
    pCrossover_size_min, pCrossover_size_max = map(float, pCrossover.split(','))

    pSubtreeMutation = request.form['pSubtreeMutation']
    pSubtreeMutation_size_min, pSubtreeMutation_size_max = map(float, pSubtreeMutation.split(','))

    pHoistMutation = request.form['pHoistMutation']
    pHoistMutation_size_min, pHoistMutation_size_max = map(float, pHoistMutation.split(','))

    pPointMutation = request.form['pPointMutation']
    pPointMutation_size_min, pPointMutation_size_max = map(float, pPointMutation.split(','))

    parsimonyCoefficient = request.form['parsimonyCoefficient']
    parsimonyCoefficient_size_min, parsimonyCoefficient_size_max = map(float, parsimonyCoefficient.split(','))

    tournamentSize = request.form['tournamentSize']
    tournamentSize_size_min, tournamentSize_size_max = map(int, tournamentSize.split(','))

    randomState = request.form['randomState']
    randomState_size_min, randomState_size_max = map(int, randomState.split(','))

    if file:
        # try:
        # 保存文件到服务器
        if not os.path.exists('file1'):
            os.makedirs('file1')
        file.save(os.path.join('file1', file.filename))

        # 读取上传的文件
        data = pd.read_excel(file)
        # X_scaler = StandardScaler()
        # y_scaler = StandardScaler()
        X_scaler = MaxAbsScaler()
        y_scaler = MaxAbsScaler()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        uni_X = X_scaler.fit_transform(X)
        # uni_y = y_scaler.fit_transform(y).ravel()
        uni_y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        dump(X_scaler, 'file1/X_scaler.joblib')
        dump(y_scaler, 'file1/y_scaler.joblib')

        function_set = request.form['functionSet'].split(',')

        param_distributions = {
            'symbolic__population_size': randint(population_size_min, population_size_max),
            'symbolic__generations': randint(generations_size_min, generations_size_max),
            'symbolic__tournament_size': randint(tournamentSize_size_min, tournamentSize_size_max),
            'symbolic__function_set': [function_set],
            'symbolic__parsimony_coefficient': uniform(parsimonyCoefficient_size_min, parsimonyCoefficient_size_max),
            'symbolic__max_samples': [maxSamples],
            'symbolic__p_crossover': uniform(pCrossover_size_min, pCrossover_size_max),
            'symbolic__p_subtree_mutation': uniform(pSubtreeMutation_size_min, pSubtreeMutation_size_max),
            'symbolic__p_hoist_mutation': uniform(pHoistMutation_size_min, pHoistMutation_size_max),
            'symbolic__p_point_mutation': uniform(pPointMutation_size_min, pPointMutation_size_max),
            'symbolic__random_state': randint(randomState_size_min, randomState_size_max)
        }
        pipe = Pipeline([
            ('symbolic', SymbolicTransformer(random_state=42)),
            ('forest', RandomForestRegressor(random_state=42))
        ])

        search = RandomizedSearchCV(pipe, param_distributions=param_distributions, n_iter=n_searches, cv=n_cv,
                                    scoring='r2', random_state=42, verbose=2, n_jobs=4)

        # 假设X和y是您的数据和目标变量
        search.fit(uni_X, uni_y)

        # 返回最佳参数和分数
        response = {
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        return jsonify(response)
    # except Exception as e:
    # return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    # return e


@app.route('/generate_features', methods=['POST'])
def generate_features():
    # 获取前端传输的 best_params
    best_params_get = request.form['best_params']
    best_params = json.loads(best_params_get)
    print(best_params)
    file = request.files['file']
    data = pd.read_excel(file)
    feature_names = data.columns[:-1].tolist()
    X_scaler = MaxAbsScaler()
    y_scaler = MaxAbsScaler()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    uni_X = X_scaler.fit_transform(X)
    uni_y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    symbolic_transformer = SymbolicTransformer(
        population_size=best_params['symbolic__population_size'],
        generations=best_params['symbolic__generations'],
        tournament_size=best_params['symbolic__tournament_size'],
        function_set=best_params['symbolic__function_set'],
        parsimony_coefficient=best_params['symbolic__parsimony_coefficient'],
        max_samples=best_params['symbolic__max_samples'],
        p_crossover=best_params['symbolic__p_crossover'],
        p_subtree_mutation=best_params['symbolic__p_subtree_mutation'],
        p_hoist_mutation=best_params['symbolic__p_hoist_mutation'],
        p_point_mutation=best_params['symbolic__p_point_mutation'],
        n_components=100,
        random_state=best_params['symbolic__random_state'],
    )
    symbolic_transformer.fit(uni_X, uni_y)

    def replace_feature_names(formula, feature_names):
        def repl(match):
            index = int(match.group(1))  # 从X0对应的是索引0开始
            return feature_names[index] if index < len(feature_names) else match.group(0)

        return re.sub(r'X(\d+)', repl, formula)

    formulas = [str(sym) for sym in symbolic_transformer._programs[-1] if sym is not None]
    print("共构建出特征个数为：", len(formulas))
    formulas_replaced = [replace_feature_names(formula, feature_names) for formula in formulas]
    feature_uni_X = pd.DataFrame(uni_X, columns=feature_names)

    # 初始化列表
    valid_formulas = []  # 有效的公式（转换后）
    valid_formulas_original = []  # 有效的公式（转换前）
    new_features = []  # 新特征值列表

    # 计算特征并检查有效性
    for i, formula in enumerate(formulas_replaced):
        try:
            new_feature_series = feature_uni_X.apply(lambda row: eval(formula, globals(), row.to_dict()), axis=1)
            if not np.isinf(new_feature_series).any():
                new_features.append(new_feature_series)  # 添加到新特征列表
                valid_formulas.append(formula)  # 保存转换后的有效公式
                valid_formulas_original.append(formulas[i])  # 保存转换前的有效公式
        except Exception as e:
            print(f"Error calculating feature {i}: {e}")

    print("符合数学计算公式的特征个数为：", len(valid_formulas))

    # 保存转换前后的公式到文本文件
    with open('file1/原始公式.txt', 'w') as file1:
        for formula in valid_formulas_original:
            file1.write(formula + '\n')

    with open('file1/特征公式.txt', 'w') as file2:
        for formula in valid_formulas:
            file2.write(formula + '\n')

    # 将新特征合并为DataFrame
    new_uni_X = pd.concat(new_features, axis=1)

    # 保存新特征到CSV文件
    new_uni_X.to_csv("file1/总-初始构建特征值.csv", index=False)

    # 使用全部新生成的特征进行随机森林模型的训练
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(new_uni_X, uni_y)  # 使用全部新生成特征进行训练
    feature_importances = rf.feature_importances_

    # 将特征重要性与特征名称关联
    feature_importance_dict = {column: importance for column, importance in zip(new_uni_X.columns, feature_importances)}

    # 根据特征重要性排序，选择顶部20个特征
    top_new_features_names = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:20]

    # 更新公式列表以匹配选择的顶部特征
    top_features_formulas = [valid_formulas[new_uni_X.columns.get_loc(name)] for name in top_new_features_names]
    top_features_formulas_original = [valid_formulas_original[new_uni_X.columns.get_loc(name)] for name in
                                      top_new_features_names]

    # 打印选定的顶部新特征、它们的随机森林特征重要性以及对应的公式
    for feature_name, formula, formula_original in zip(top_new_features_names, top_features_formulas,
                                                       top_features_formulas_original):
        importance = feature_importance_dict[feature_name]
        print(
            f"Feature: {feature_name}, Importance: {importance:.4f}, Formula: {formula}, Original Formula: {formula_original}")

    # 获取顶部新特征的DataFrame
    top_new_features_df = new_uni_X[top_new_features_names]

    # 导出最终的特征集到CSV文件
    top_new_features_df.to_csv("file1/final_top_features.csv", index=False)

    return jsonify({
        'featureNum': len(formulas),
        'featureMathNum': len(valid_formulas),
        'featureInfo': f"Feature: {feature_name}, Importance: {importance:.4f}, Formula: {formula}, Original Formula: {formula_original}"
    }), 200


@app.route('/finance_symbolic_regression', methods=['POST'])
def finance_symbolic_regression():
    if 'file' not in request.files:
        print(request)
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    n_cv = int(request.form['cv'])
    n_searches = int(request.form['n_searches'])
    population_size = request.form['population_size']
    population_size_min, population_size_max = map(int, population_size.split(','))

    generations = request.form['generations']
    generations_size_min, generations_size_max = map(int, generations.split(','))

    maxSamples = int(request.form['maxSamples'])

    pCrossover = request.form['pCrossover']
    pCrossover_size_min, pCrossover_size_max = map(float, pCrossover.split(','))

    pSubtreeMutation = request.form['pSubtreeMutation']
    pSubtreeMutation_size_min, pSubtreeMutation_size_max = map(float, pSubtreeMutation.split(','))

    pHoistMutation = request.form['pHoistMutation']
    pHoistMutation_size_min, pHoistMutation_size_max = map(float, pHoistMutation.split(','))

    pPointMutation = request.form['pPointMutation']
    pPointMutation_size_min, pPointMutation_size_max = map(float, pPointMutation.split(','))

    parsimonyCoefficient = request.form['parsimonyCoefficient']
    parsimonyCoefficient_size_min, parsimonyCoefficient_size_max = map(float, parsimonyCoefficient.split(','))

    randomState = request.form['randomState']
    randomState_size_min, randomState_size_max = map(int, randomState.split(','))

    if file:
        # try:
        # 保存文件到服务器
        if not os.path.exists('finance'):
            os.makedirs('finance')
        file.save(os.path.join('finance', file.filename))

        # 读取上传的文件
        data = pd.read_excel(file)
        function_set = request.form['functionSet'].split(',')
        X_scaler = MaxAbsScaler()
        y_scaler = MaxAbsScaler()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values.reshape((-1, 1))
        uni_X = X_scaler.fit_transform(X)
        uni_y = y_scaler.fit_transform(y).ravel()
        best_score = -np.inf
        best_params = None

        for _ in tqdm(range(n_searches), desc="Searching"):

            random_state = randint(randomState_size_min, randomState_size_max).rvs()

            params = {
                'population_size': randint(population_size_min, population_size_max).rvs(random_state=random_state),
                'generations': randint(generations_size_min, generations_size_max).rvs(random_state=random_state),
                'parsimony_coefficient': uniform(parsimonyCoefficient_size_min, parsimonyCoefficient_size_max).rvs(
                    random_state=random_state),
                'max_samples': maxSamples,
                'function_set': function_set,
                'p_crossover': uniform(pCrossover_size_min, pCrossover_size_max).rvs(random_state=random_state),
                'p_subtree_mutation': uniform(pSubtreeMutation_size_min, pSubtreeMutation_size_max).rvs(
                    random_state=random_state),
                'p_hoist_mutation': uniform(pHoistMutation_size_min, pHoistMutation_size_max).rvs(
                    random_state=random_state),
                'p_point_mutation': uniform(pPointMutation_size_min, pPointMutation_size_max).rvs(
                    random_state=random_state),
            }

            # 创建模型
            model = SymbolicRegressor(random_state=random_state, **params)

            # 评估模型
            score = np.mean(cross_val_score(model, uni_X, uni_y, cv=n_cv))

            params['random_state'] = random_state
            # 更新最佳模型
            if score > best_score:
                best_score = score
                best_params = params

        # 打印最佳参数和分数
        print("Best Score:", best_score)
        print("Best Parameters:", best_params)
        response = {
            'best_score': best_score,
            'best_params': best_params
        }
        return jsonify(response)
    # except Exception as e:
    # return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    # return e


@app.route('/generate_finance_features', methods=['POST'])
def generate_finance_features():
    file = request.files['file']
    data = pd.read_excel(file).values
    X_scaler = MaxAbsScaler()  # 修改处
    y_scaler = MaxAbsScaler()  # 修改处

    X = data[:, :-1]
    y = data[:, -1]
    y = y.reshape((-1, 1))

    uni_X = X_scaler.fit_transform(X)
    uni_y = y_scaler.fit_transform(y).ravel()
    # 获取前端传输的 best_params
    best_params_get = request.form['best_params']
    best_params = json.loads(best_params_get)
    best_model = SymbolicRegressor(
        generations=best_params['generations'],  # 进化代数
        population_size=best_params['population_size'],  # 初始化树分支数量
        function_set=best_params['function_set'],  # 符号范围
        parsimony_coefficient=best_params['parsimony_coefficient'],  # 简化系数，控制公式复杂度
        max_samples=best_params['max_samples'],
        p_crossover=best_params['p_crossover'],
        p_subtree_mutation=best_params['p_subtree_mutation'],
        p_hoist_mutation=best_params['p_hoist_mutation'],
        p_point_mutation=best_params['p_point_mutation'],
        verbose=1,
        random_state=best_params['random_state']
    )
    best_model.fit(uni_X, uni_y)
    best_formula = best_model._program

    # 打印最佳公式
    print("Best Formula:", best_formula)
    return jsonify({
        'BestFormula': str(best_formula)

    }), 200
    # return jsonify({
    #     'featureNum': len(formulas),
    #     'featureMathNum': len(valid_formulas),
    #     'featureInfo': f"Feature: {feature_name}, Importance: {importance:.4f}, Formula: {formula}, Original Formula: {formula_original}"
    # }), 200


# @app.route('/get_gp_features', methods=['POST'])
# def get_gp_features():
#     # 获取前端传输的 best_params
#     best_params_get = request.form['best_params']
#     best_params = json.loads(best_params_get)
#     file = request.files['file']
#
#     # 使用前端传输的 best_params 创建 SymbolicTransformer
#     gp = SymbolicTransformer(
#         generations=best_params['generations'],
#         population_size=best_params['population_size'],
#         hall_of_fame=20,
#         n_components=20,
#         function_set=best_params['function_set'],
#         parsimony_coefficient=best_params['parsimony_coefficient'],
#         max_samples=best_params['max_samples'],
#         p_crossover=best_params['p_crossover'],
#         p_subtree_mutation=best_params['p_subtree_mutation'],
#         p_hoist_mutation=best_params['p_hoist_mutation'],
#         p_point_mutation=best_params['p_point_mutation'],
#         verbose=1,
#         random_state=best_params['random_state']
#     )
#     data = pd.read_excel(file)  # 读取为DataFrame
#     feature_names = data.columns[:-1].tolist()  # 获取特征名
#     X = data.iloc[:, :-1].values  # 提取特征矩阵
#     y = data.iloc[:, -1].values.reshape(-1, 1)  # 提取目标变量并进行形状调整
#
#     X_scaler = StandardScaler()
#     y_scaler = StandardScaler()
#     uni_X = X_scaler.fit_transform(X)
#     uni_y = y_scaler.fit_transform(y).ravel()
#     gp.fit(uni_X, uni_y)
#
#     # 使用 gp.fit_transform 获得新特征
#     gp_features = gp.transform(uni_X)
#
#     # 将新特征存储到 DataFrame 中
#     gp_df = pd.DataFrame(gp_features, columns=[f'gpf{i}' for i in range(gp_features.shape[1])])
#
#     # 将 DataFrame 存储为 CSV 文件
#     csv_filename = f"gp_features.csv"
#     gp_df.to_csv(csv_filename, index=False)
#
#     # 保存替换前的公式
#     filename0 = 'symbolic_transformer_formulas.txt'
#     with open(filename0, 'w') as file:
#         for formula in gp._programs[-1]:
#             file.write(str(formula) + '\n')
#     print(f'特征名替换前的公式已保存到文件：{filename0}')
#
#     # 替换特征名的函数
#     def replace_feature_names(formula, feature_names):
#         def repl(match):
#             index = int(match.group(1))  # 从X1对应的是索引0开始
#             return feature_names[index] if index < len(feature_names) else match.group(0)
#
#         return re.sub(r'X(\d+)', repl, str(formula))
#
#     # 使用特征名替换
#     formulas_replaced = [replace_feature_names(formula, feature_names) for formula in gp._programs[-1]]
#
#     # 保存替换后的公式
#     filename1 = 'symbolic_transformer_formulas_with_feature_names.txt'
#     with open(filename1, 'w') as file:
#         for formula in formulas_replaced:
#             file.write(formula + '\n')
#
#     print(f'特征名替换后的公式已保存到文件：{filename1}')
#
#     return jsonify({'csv_filename': csv_filename, 'formula_txt': filename0, 'formula_with_names_txt': filename1})

@app.route('/download_file1_x_job', methods=['GET'])
def download_file1_x_job():
    model_path = 'file1/X_scaler.joblib'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_file1_y_job', methods=['GET'])
def download_file1_y_job():
    model_path = 'file1/y_scaler.joblib'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_final_csv', methods=['GET'])
def download_final_csv():
    model_path = 'file1/final_top_features.csv'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_total_csv', methods=['GET'])
def download_total_csv():
    model_path = 'file1/总-初始构建特征值.csv'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_txt', methods=['GET'])
def download_txt():
    model_path = 'file1/原始公式.txt'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_txt_names', methods=['GET'])
def download_txt_names():
    model_path = 'file1/特征公式.txt'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/feature_selection', methods=['POST'])
def feature_selection():
    def evaluate_model_with_subset(model, feature_subset, X_train, y_train, X_test, y_test):
        try:
            model.fit(X_train[:, feature_subset], y_train)
            y_pred = model.predict(X_test[:, feature_subset])
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return mse, r2, feature_subset

        except Exception as e:
            print(f"Error in evaluating model with subset {feature_subset}: {e}")
            return float('inf'), -1, feature_subset

    try:
        file = request.files['file']
        filename = file.filename

        # 保存上传的文件
        file.save(filename)

        # 读取上传的文件数据
        data = pd.read_excel(filename)
        # 检查数据列名是否与特征名称匹配

        X = data.iloc[:, :-1].values  # 转换为numpy数组
        y = data.iloc[:, -1].values  # 转换为numpy数组

        # 初始化标准化器
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # 对X进行标准化
        uni_X = X_scaler.fit_transform(X)
        # 对y进行标准化
        uni_y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(uni_X, uni_y, test_size=0.3, random_state=42)

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

        # 对每种模型独立进行特征选择
        for model_name, model in models.items():
            model_results = []

            for subset_size in range(1, len(data.columns)):
                all_subsets = list(combinations(range(len(data.columns) - 1), subset_size))

                results = Parallel(n_jobs=-1)(
                    delayed(evaluate_model_with_subset)(model, subset, X_train, y_train, X_test, y_test)
                    for subset in all_subsets
                )

                model_results.extend(results)

            # 对所有结果排序并获取前10个
            top_10_results = sorted(model_results, key=lambda x: x[0])[:10]

            # 将前10名结果添加到DataFrame中
            for mse, r2, subset in top_10_results:
                new_row = pd.DataFrame(
                    {
                        'Model': [model_name],
                        'Feature Set': [', '.join(data.columns[list(subset)])],  # 将特征索引转换为特征名称
                        'MSE': [mse],
                        'R2': [r2]
                    }
                )
            new_row = new_row.dropna(axis=1, how='all')
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        if not os.path.exists('result_files'):
            os.makedirs('result_files')
        # 生成结果文件并保存在服务器上
        result_filename = f"results.csv"
        results_df.to_csv(os.path.join('result_files', result_filename), index=False)

        return jsonify({'message': 'successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


class ANN(nn.Module):
    def __init__(self, input_features):
        super(ANN, self).__init__()
        # 根据输入特征数量动态调整
        self.fc1 = nn.Sequential(nn.Linear(input_features, 128),
                                 nn.Dropout(0.3),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(128, 64),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 32),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(32, 16),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU())
        self.fc5 = nn.Sequential(nn.Linear(16, 8),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU())
        self.fc6 = nn.Sequential(nn.Linear(8, 1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x


@app.route('/train', methods=['POST'])
def train_model():
    if 'data' not in request.files or 'data2' not in request.files:
        return jsonify({'error': 'No files found'}), 400
    if not os.path.exists('train'):
        os.makedirs('train')

    def data_division(uni_X, uni_y):
        X_train, X_test, y_train, y_test = train_test_split(uni_X, uni_y, test_size=0.3)
        return X_train, X_test, y_train, y_test

    def make_loader(X_train, X_test, y_train, y_test):
        X_train = torch.from_numpy(X_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)

        batch_sizes = 32
        train_dataset = Data.TensorDataset(X_train, y_train)
        test_dataset = Data.TensorDataset(X_test, y_test)
        train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_sizes, shuffle=True)
        test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_sizes, shuffle=True)

        return train_dataloader, test_dataloader

    def train(train_dataloader):
        # 训练状态
        model.train()
        running_loss = 0.0
        samples_nums = 0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data  # 取出X和Y
            out = model(inputs)  # 前向预测

            loss = loss_fn(out, labels)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # Loss反向传播
            optimizer.step()  # 优化器优化

            running_loss += loss.item() * inputs.size(0)
            samples_nums += inputs.size(0)

        epoch_loss = running_loss / samples_nums  # 平均loss
        return epoch_loss

    def test(test_dataloader):
        # 训练状态
        model.eval()
        running_loss = 0.0
        samples_nums = 0
        for i, data in enumerate(test_dataloader):
            # data中有部分数据需要处理
            inputs, labels = data  # 取出X和Y
            # 预测过程
            out = model(inputs)  # 前向预测
            loss = loss_fn(out, labels)  # 计算Loss
            running_loss += loss.item() * inputs.size(0)
            samples_nums += inputs.size(0)

        epoch_loss = running_loss / samples_nums  # 平均loss

        return epoch_loss

    def evaluate_model(model, test_dataloader):
        model.eval()  # 将模型设置为评估模式
        predictions = []
        actuals = []
        with torch.no_grad():  # 在评估阶段禁用梯度计算
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)  # 确保数据在正确的设备上
                y_pred = model(X)  # 获取模型预测
                predictions.extend(y_pred.cpu().numpy())  # 存储预测值
                actuals.extend(y.cpu().numpy())  # 存储实际值

        return np.array(predictions), np.array(actuals)

    file1 = request.files['data']
    file2 = request.files['data2']
    data = pd.read_excel(file1).values
    X_scaler = StandardScaler()  # StandardScaler() #
    y_scaler = StandardScaler()  # StandardScaler() #

    filename = file1.filename
    X = data[:, :-1]
    # 需要输入之前得到的那个文件
    data2 = pd.read_csv(file2).values
    cycles = int(request.form.get('cycles'))
    epoch = int(request.form.get('epoch'))
    feature_add_num = int(request.form.get('feature_add_num'))
    model = ANN(X.shape[1] + feature_add_num).to(device)
    if feature_add_num != 0:
        X = np.concatenate([X, data2[:, 0:feature_add_num]], axis=1)

    y = data[:, -1]
    y = y.reshape((-1, 1))

    uni_X = X_scaler.fit_transform(X)
    uni_y = y_scaler.fit_transform(y)
    dump(X_scaler, 'train/X_scaler.joblib')
    dump(y_scaler, 'train/y_scaler.joblib')
    best_loss = float('inf')
    best_model_state = None
    best_cycle = -1

    results = []
    for cycle_i in range(0, cycles):
        clear_output()

        # 权重初始化
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # 训练参数
        LR = 0.001
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), LR)
        epochs = epoch

        # 数据集划分
        X_train, X_test, y_train, y_test = data_division(uni_X, uni_y)
        train_dataloader, test_dataloader = make_loader(X_train, X_test, y_train, y_test)

        train_loss = np.zeros((epochs, 1))
        test_loss = np.zeros((epochs, 1))
        milestones = [300, 2000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for epoch in range(epochs):
            print('epoch:', epoch)
            train_loss[epoch] = train(train_dataloader)
            test_loss[epoch] = test(test_dataloader)
            scheduler.step()

        pre_test_y, exp_test_y = evaluate_model(model, test_dataloader)
        r2 = r2_score(exp_test_y, pre_test_y)
        mse = mean_squared_error(exp_test_y, pre_test_y)

        # 创建 DataFrame 并保存当前循环的结果
        cycle_result = pd.DataFrame([[filename, feature_add_num, cycle_i, r2, mse]],
                                    columns=['Filename', 'Feature_Add_Num', 'Cycle', 'R2', 'MSE'])

        csv_filename = f'train/results.csv'
        directory = os.path.dirname(csv_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 如果文件不存在，则创建新文件，否则追加到现有文件
        if not os.path.isfile(csv_filename):
            cycle_result.to_csv(csv_filename, index=False, mode='w')
        else:
            cycle_result.to_csv(csv_filename, index=False, mode='a', header=False)

        print(f'Results for {filename}, feature {feature_add_num}, cycle {cycle_i} saved to {csv_filename}')

        # 检查当前模型是否是最佳模型
        current_test_loss = test_loss[-1, 0]
        if current_test_loss < best_loss:
            best_loss = current_test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_cycle = cycle_i
            print(f"Found a better model at cycle {cycle_i} with test loss: {best_loss}")

    if best_model_state is not None:
        torch.save(best_model_state, f'train/bestModel.pth')
        print("Best model saved with test loss:", best_loss, "in cycle", best_cycle)
    else:
        print("No model improved.")
    return jsonify({'message': 'Model training successful'}), 200


@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = 'train/bestModel.pth'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_feature_set', methods=['GET'])
def download_feature_set():
    model_path = 'result_files/results.csv'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


# Endpoint to return the results CSV file
@app.route('/download_results', methods=['GET'])
def download_results():
    temp_csv_path = 'train/results.csv'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_joblibX', methods=['GET'])
def download_joblib_x():
    temp_csv_path = 'train/X_scaler.joblib'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_joblibY', methods=['GET'])
def download_joblib_y():
    temp_csv_path = 'train/y_scaler.joblib'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_predict_file', methods=['GET'])
def download_predict_file():
    temp_csv_path = 'predict/combined_input_and_prediction.xlsx'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    # try:
    # 获取上传的文件
    if not os.path.exists('predict'):
        os.makedirs('predict')
    model_file = request.files['model']  # .pth
    file = request.files['data_pre']
    X_scaler = load(request.files['x_joblib'])
    y_scaler = load(request.files['y_joblib'])
    tzTxt = request.files['tzTxt']
    tzTxt.save(os.path.join('predict', tzTxt.filename))
    # 读取保存的文件
    data_pre = pd.read_excel(file)
    feature_add_num = 0  # 输入添加特征数量
    i = 0
    # 初始化已成功添加的特征数量
    added_features_count = 0
    with open(os.path.join('predict', tzTxt.filename), 'r') as file:
        formulas_with_names = file.readlines()

    formulas_with_names = [line.strip() for line in formulas_with_names]

    pre_data_scaled = pd.DataFrame(X_scaler.transform(data_pre), columns=data_pre.columns)

    while added_features_count < feature_add_num and i < len(formulas_with_names):
        formula = formulas_with_names[i]
        try:
            # 尝试根据公式添加新特征
            pre_data_scaled[f'new_feature_{added_features_count}'] = pre_data_scaled.apply(
                lambda row: eval(formula, globals(), row.to_dict()), axis=1)
            # 如果成功，则增加已添加特征的计数
            added_features_count += 1
        except Exception as e:
            # 如果出错，打印错误消息但不增加已添加特征的计数
            print(f"Error calculating feature {i}: {e}")
        # 无论成功与否，都移动到下一个公式
        i += 1

    input_tensor = torch.from_numpy(pre_data_scaled.values).float().to(device)
    print(f"最终成功添加的新特征数量：{added_features_count}")
    model = ANN(pre_data_scaled.shape[1]).to(device)
    model.load_state_dict(
        # torch.load(f"E:/LBE/result/1000_noSarOinldata - 副本_best_model_0.pth", map_location=device))
        torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor)

    predicted_output = prediction.cpu().numpy()
    predicted_output_scaled_back = y_scaler.inverse_transform(predicted_output)

    xx = pd.DataFrame(predicted_output_scaled_back)
    # 确保预测结果是DataFrame，并且具有合适的列名
    xx.columns = ['Predicted_Value']  # 为预测结果设置列名，确保这个名字不与输入数据的任何列名冲突

    # 拼接输入数据与预测结果
    result_df = pd.concat([data_pre, xx], axis=1)  # data_pre 是原始的、未经标准化处理的输入数据

    # 保存到Excel
    file_path = 'predict/combined_input_and_prediction.xlsx'  # 指定文件路径和名称
    result_df.to_excel(file_path, index=True)
    # ttttt
    return jsonify({'res': 'successful'}), 200


# except Exception as e:
#     return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/serverConfig.json', methods=['GET'])
def config():
    return send_file('templates/serverConfig.json')


@app.route('/assets/<path:filename>', methods=['GET'])
def assets(filename):
    return send_from_directory('templates/assets', filename)


@app.route('/static/<path:pathf>/<path:filename>', methods=['GET'])
def staticjs(filename, pathf):
    return send_from_directory('templates/static', pathf + '/' + filename)


if __name__ == '__main__':
    app.run(debug=True)
