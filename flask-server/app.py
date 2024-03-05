import copy
import json
import os
import re
import tempfile

from flask import Flask, jsonify, request, send_file, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from joblib import Parallel, delayed, dump, load
from torch import nn, optim, from_numpy
import torch
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from gplearn.genetic import SymbolicTransformer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 设置文件上传目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 设置临时文件目录
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'pth'}

uploaded_data = None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_data
    # 检查是否存在上传文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 检查文件名是否合法
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'}), 400

    # 保存文件到临时文件夹
    filename = secure_filename(file.filename)
    temp_filepath = os.path.join(TEMP_FOLDER, filename)
    file.save(temp_filepath)

    # 读取文件内容
    if filename.endswith('.csv'):
        data = pd.read_csv(temp_filepath).values
        uploaded_data = {
            'X': data[:, :-1],
            'y': data[:, -1].reshape((-1, 1))
        }
    elif filename.endswith('.xlsx'):
        data = pd.read_excel(temp_filepath, engine='openpyxl').values
        uploaded_data = {
            'X': data[:, :-1],
            'y': data[:, -1].reshape((-1, 1))
        }
    elif filename.endswith('.pth'):
        pass
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    # 返回临时文件路径
    return jsonify({'message': 'File uploaded successfully', 'temp_file_path': temp_filepath})


@app.route('/file1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print(request)
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
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

    functionSet = request.form['functionSet']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        try:
            # 保存文件到服务器
            print(uploaded_data)
            file.save(file.filename)

            # 读取上传的文件
            data = pd.read_excel(file.filename).values
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X = data[:, :-1]
            y = data[:, -1].reshape((-1, 1))

            uni_X = X_scaler.fit_transform(X)
            uni_y = y_scaler.fit_transform(y).ravel()

            function_set = request.form['functionSet'].split(',')
            print(function_set)

            # param_range = {
            #     'population_size': (1000, 10000),
            #     'generations': (10, 50),
            #     'parsimony_coefficient': (0.00000000001, 0.0001),
            # }

            best_score = -np.inf
            best_params = None

            for _ in tqdm(range(n_searches), desc="Searching"):
                random_state = randint(1, 1000).rvs()

                print(population_size_min, population_size_max, 'adsasdasdasdasdasdasdad')
                params = {
                    # 'population_size': randint(*param_range['population_size']).rvs(random_state=random_state),
                    # 'generations': randint(*param_range['generations']).rvs(random_state=random_state),
                    # 'parsimony_coefficient': uniform(*param_range['parsimony_coefficient']).rvs(
                    #     random_state=random_state),
                    'population_size': int(
                        uniform(population_size_min, population_size_max).rvs(random_state=random_state)),
                    'generations': int(
                        uniform(generations_size_min, generations_size_max).rvs(random_state=random_state)),
                    'parsimony_coefficient': uniform(parsimonyCoefficient_size_min, parsimonyCoefficient_size_max).rvs(
                        random_state=random_state),
                    'max_samples': maxSamples,
                    'function_set': function_set
                }
                initial_probabilities = {
                    'p_crossover': uniform(pCrossover_size_min, pCrossover_size_max).rvs(random_state=random_state),
                    # 调整为在0.3到0.5之间
                    'p_subtree_mutation': uniform(pSubtreeMutation_size_min, pSubtreeMutation_size_max).rvs(
                        random_state=random_state),  # 调整为在0.1到0.25之间
                    'p_hoist_mutation': uniform(pHoistMutation_size_min, pHoistMutation_size_max).rvs(
                        random_state=random_state),  # 调整为在0.05到0.15之间
                    'p_point_mutation': uniform(pPointMutation_size_min, pPointMutation_size_max).rvs(
                        random_state=random_state + 1),  # 调整为在0.05到0.15之间
                }

                # 确保总和小于等于1
                total_probability = sum(initial_probabilities.values())
                if total_probability > 1:
                    normalized_probabilities = {k: v / total_probability for k, v in initial_probabilities.items()}
                else:
                    normalized_probabilities = initial_probabilities

                params.update(normalized_probabilities)

                model = SymbolicRegressor(random_state=random_state, **params)

                score = np.mean(cross_val_score(model, uni_X, uni_y, cv=3))

                params['random_state'] = random_state
                if score > best_score:
                    best_score = score
                    best_params = params

            # 返回最佳参数和分数
            response = {
                'best_score': best_score,
                'best_params': best_params
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/generate_features', methods=['POST'])
def generate_features():
    global uploaded_data

    # 获取前端传输的 best_params
    best_params = request.json.get('best_params')

    # 使用前端传输的 best_params 创建 SymbolicTransformer
    gp = SymbolicTransformer(
        generations=best_params['generations'],
        population_size=best_params['population_size'],
        hall_of_fame=20,
        n_components=20,
        function_set=best_params['function_set'],
        parsimony_coefficient=best_params['parsimony_coefficient'],
        max_samples=best_params['max_samples'],
        p_crossover=best_params['p_crossover'],
        p_subtree_mutation=best_params['p_subtree_mutation'],
        p_hoist_mutation=best_params['p_hoist_mutation'],
        p_point_mutation=best_params['p_point_mutation'],
        verbose=1,
        random_state=best_params['random_state']
    )
    X = uploaded_data['X']
    y = uploaded_data['y']
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    uni_X = X_scaler.fit_transform(X)
    uni_y = y_scaler.fit_transform(y).ravel()
    gp.fit(uni_X, uni_y)
    # 使用 gp.fit_transform 获得新特征
    print(uploaded_data, 'dadadad')

    gp_features = gp.transform(X)
    print(gp_features)
    # 将新特征存储到 DataFrame 中
    # dataframe = pd.DataFrame(gp_features, columns=[f'gpf{i}' for i in range(gp_features.shape[1])])
    dataframe = pd.DataFrame({
        'gpf0': gp_features[:, 0],
        'gpf1': gp_features[:, 1],
        'gpf2': gp_features[:, 2],
        'gpf3': gp_features[:, 3],
        'gpf4': gp_features[:, 4],
        'gpf5': gp_features[:, 5],
        'gpf6': gp_features[:, 6],
        'gpf7': gp_features[:, 7],
        'gpf8': gp_features[:, 8],
        'gpf9': gp_features[:, 9],
    })
    # 将 DataFrame 存储为 CSV 文件
    csv_filename = f"gp_features.csv"
    dataframe.to_csv(csv_filename, index=False)

    # 返回生成的 CSV 文件供前端下载
    return send_file(csv_filename, as_attachment=True)


@app.route('/get_gp_features', methods=['POST'])
def get_gp_features():
    global uploaded_data

    # 获取前端传输的 best_params
    best_params_get = request.form['best_params']
    best_params = json.loads(best_params_get)
    file = request.files['file']

    # 使用前端传输的 best_params 创建 SymbolicTransformer
    gp = SymbolicTransformer(
        generations=best_params['generations'],
        population_size=best_params['population_size'],
        hall_of_fame=20,
        n_components=20,
        function_set=best_params['function_set'],
        parsimony_coefficient=best_params['parsimony_coefficient'],
        max_samples=best_params['max_samples'],
        p_crossover=best_params['p_crossover'],
        p_subtree_mutation=best_params['p_subtree_mutation'],
        p_hoist_mutation=best_params['p_hoist_mutation'],
        p_point_mutation=best_params['p_point_mutation'],
        verbose=1,
        random_state=best_params['random_state']
    )
    data = pd.read_excel(file)  # 读取为DataFrame
    feature_names = data.columns[:-1].tolist()  # 获取特征名
    X = data.iloc[:, :-1].values  # 提取特征矩阵
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 提取目标变量并进行形状调整

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    uni_X = X_scaler.fit_transform(X)
    uni_y = y_scaler.fit_transform(y).ravel()
    gp.fit(uni_X, uni_y)

    # 使用 gp.fit_transform 获得新特征
    gp_features = gp.transform(uni_X)

    # 将新特征存储到 DataFrame 中
    gp_df = pd.DataFrame(gp_features, columns=[f'gpf{i}' for i in range(gp_features.shape[1])])

    # 将 DataFrame 存储为 CSV 文件
    csv_filename = f"gp_features.csv"
    gp_df.to_csv(csv_filename, index=False)

    # 保存替换前的公式
    filename0 = 'symbolic_transformer_formulas.txt'
    with open(filename0, 'w') as file:
        for formula in gp._programs[-1]:
            file.write(str(formula) + '\n')
    print(f'特征名替换前的公式已保存到文件：{filename0}')

    # 替换特征名的函数
    def replace_feature_names(formula, feature_names):
        def repl(match):
            index = int(match.group(1))  # 从X1对应的是索引0开始
            return feature_names[index] if index < len(feature_names) else match.group(0)

        return re.sub(r'X(\d+)', repl, str(formula))

    # 使用特征名替换
    formulas_replaced = [replace_feature_names(formula, feature_names) for formula in gp._programs[-1]]

    # 保存替换后的公式
    filename1 = 'symbolic_transformer_formulas_with_feature_names.txt'
    with open(filename1, 'w') as file:
        for formula in formulas_replaced:
            file.write(formula + '\n')

    print(f'特征名替换后的公式已保存到文件：{filename1}')

    return jsonify({'csv_filename': csv_filename, 'formula_txt': filename0, 'formula_with_names_txt': filename1})


@app.route('/download_txt', methods=['GET'])
def download_txt():
    model_path = 'symbolic_transformer_formulas.txt'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


@app.route('/download_txt_names', methods=['GET'])
def download_txt_names():
    model_path = 'symbolic_transformer_formulas_with_feature_names.txt'
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
        print(data, 'ccccccccccccccccccccccccccccccccccccccccc')

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
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # 生成结果文件并返回给前端
        result_filename = f"{os.path.splitext(filename)[0]}_results.csv"
        results_df.to_csv(result_filename, index=False)

        # 删除上传的文件
        os.remove(filename)

        return send_file(result_filename, as_attachment=True)

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


# 数据集划分函数
def data_division(uni_X, uni_y):
    X_train, X_test, y_train, y_test = train_test_split(uni_X, uni_y, test_size=0.3)
    return X_train, X_test, y_train, y_test


# 数据加载函数
def make_loader(X_train, X_test, y_train, y_test):
    X_train = torch.from_numpy(X_train).float().to()
    X_test = torch.from_numpy(X_test).float().to()
    y_train = torch.from_numpy(y_train).float().to()
    y_test = torch.from_numpy(y_test).float().to()

    batch_sizes = 32
    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_sizes, shuffle=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_sizes, shuffle=True)

    return train_dataloader, test_dataloader


# 训练函数
def train(model, train_dataloader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        samples_nums = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            out = model(inputs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            samples_nums += inputs.size(0)
        epoch_loss = running_loss / samples_nums
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))


# 测试函数
def test(model, test_dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    samples_nums = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            out = model(inputs)
            loss = loss_fn(out, labels)
            running_loss += loss.item() * inputs.size(0)
            samples_nums += inputs.size(0)
    epoch_loss = running_loss / samples_nums
    return epoch_loss


# 评估函数
def evaluate_model(model, test_dataloader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(), y.to()
            y_pred = model(X)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return np.array(predictions), np.array(actuals)


@app.route('/train', methods=['POST'])
def train_model():
    if 'data' not in request.files or 'data2' not in request.files:
        return jsonify({'error': 'No files found'}), 400
    data = request.files['data']
    data2 = request.files['data2']
    cycles = int(request.form.get('cycles'))
    epoch = int(request.form.get('epoch'))
    feature_add_num = int(request.form.get('feature_add_num'))
    if data.filename == '' or data2.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    if data and allowed_file(data.filename) and data2 and allowed_file(data2.filename):
        data1 = pd.read_excel(data) if data.filename.endswith('.xlsx') else pd.read_csv(data, encoding='gbk')
        data2_df = pd.read_csv(data2) if data2.filename.endswith('.csv') else pd.read_excel(data2)
        X1 = data1.iloc[:, :-1].values
        X2 = data2_df.iloc[:, :feature_add_num].values
        y = data1.iloc[:, -1].values.reshape(-1, 1)
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        uni_X = np.concatenate((X1, X2), axis=1)
        uni_X = X_scaler.fit_transform(uni_X)  # Standardize the combined features
        uni_y = y_scaler.fit_transform(y)  # Standardize the target variable
        print(uni_X.shape)
        dump(X_scaler, 'X_scaler.joblib')
        dump(y_scaler, 'y_scaler.joblib')
        model = ANN(uni_X.shape[1]).to()
        X_train, X_test, y_train, y_test = data_division(uni_X, uni_y)
        train_dataloader, test_dataloader = make_loader(X_train, X_test, y_train, y_test)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_loss = float('inf')
        best_model_state = None
        results = []
        milestones = [300, 2000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        for cycle_i in range(cycles):
            train_loss = train(model, train_dataloader, loss_fn, optimizer, epoch)
            test_loss = test(model, test_dataloader, loss_fn)
            scheduler.step()

            pre_test_y, exp_test_y = evaluate_model(model, test_dataloader)
            r2 = r2_score(exp_test_y, pre_test_y)
            mse = mean_squared_error(exp_test_y, pre_test_y)

            # Save results for this cycle
            cycle_result = {'Filename': data.filename, 'Feature_Add_Num': feature_add_num,
                            'Cycle': cycle_i, 'R2': r2, 'MSE': mse}
            results.append(cycle_result)

            current_test_loss = test_loss
            if current_test_loss < best_loss:
                best_loss = current_test_loss
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state is not None:
            model_path = f'models/best_model.pth'
            torch.save(best_model_state, model_path)

            # Save results to CSV file
            results_df = pd.DataFrame(results)
            results_csv_path = 'results_data.csv'
            results_df.to_csv(results_csv_path, index=False)

            # Save data2 to a temporary CSV file
            temp_csv_path = 'temp_data2.csv'
            data2_df.to_csv(temp_csv_path, index=False)

            # Return both model file and CSV files
            return send_file(results_csv_path, as_attachment=True)
        else:
            return jsonify({'error': 'Failed to train model'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404


# Endpoint to return the results CSV file
@app.route('/download_results', methods=['GET'])
def download_results():
    temp_csv_path = 'results_data.csv'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_joblibX', methods=['GET'])
def download_joblib_x():
    temp_csv_path = 'X_scaler.joblib'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_joblibY', methods=['GET'])
def download_joblib_y():
    temp_csv_path = 'y_scaler.joblib'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/download_predict_file', methods=['GET'])
def download_predict_file():
    temp_csv_path = 'predicted_output.xlsx'
    if os.path.exists(temp_csv_path):
        return send_file(temp_csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'Results file not found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    X_scaler = load('X_scaler.joblib')
    y_scaler = load('y_scaler.joblib')
    # 获取上传的文件
    model_file = request.files['model']
    data_pre_file = request.files['data_pre']

    # 读取预测数据
    data_pre = pd.read_excel(data_pre_file)
    data_pre = data_pre.values
    print(data_pre.shape)
    # 数据预处理
    X_pre_scaled = X_scaler.transform(data_pre)
    input_tensor = torch.from_numpy(X_pre_scaled).float()

    # 加载模型并进行推理
    model = ANN(data_pre.shape[1])
    model.load_state_dict(torch.load(model_file))
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    # 将预测结果反向转换并保存为Excel文件
    predicted_output = prediction.cpu().numpy()
    predicted_output_scaled_back = y_scaler.inverse_transform(predicted_output)
    xx = pd.DataFrame(predicted_output_scaled_back)
    file_path = 'predicted_output.xlsx'
    xx.to_excel(file_path, index=True)

    return jsonify({'file_path': file_path})


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
