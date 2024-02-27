import copy
import os
import tempfile

from flask import Flask, jsonify, request, send_file
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
from joblib import Parallel, delayed
from torch import nn, optim, from_numpy
import torch
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from gplearn.genetic import SymbolicTransformer
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.X = None
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
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

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
    elif filename.endswith('.xlsx'):
        data = pd.read_excel(temp_filepath, engine='openpyxl').values
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    uploaded_data = {
        'X': data[:, :-1],
        'y': data[:, -1].reshape((-1, 1))
    }
    # 返回临时文件路径
    return jsonify({'message': 'File uploaded successfully', 'temp_file_path': temp_filepath})


@app.route('/file1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print(request)
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    n_searches = int(request.form['n_searches'])
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

            function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']

            param_range = {
                'population_size': (1000, 10000),
                'generations': (10, 50),
                'parsimony_coefficient': (0.00000000001, 0.0001),
            }

            best_score = -np.inf
            best_params = None

            for _ in tqdm(range(n_searches), desc="Searching"):
                random_state = randint(1, 1000).rvs()

                params = {
                    'population_size': randint(*param_range['population_size']).rvs(random_state=random_state),
                    'generations': randint(*param_range['generations']).rvs(random_state=random_state),
                    'parsimony_coefficient': uniform(*param_range['parsimony_coefficient']).rvs(
                        random_state=random_state),
                    'max_samples': uniform(0.8, 0.2).rvs(random_state=random_state),
                    'function_set': function_set
                }

                initial_probabilities = {
                    'p_crossover': uniform(0.1, 0.3).rvs(random_state=random_state),
                    'p_subtree_mutation': uniform(0.1, 0.3).rvs(random_state=random_state),
                    'p_hoist_mutation': uniform(0.05, 0.2).rvs(random_state=random_state),
                    'p_point_mutation': uniform(0.05, 0.2).rvs(random_state=random_state),
                }

                total_probability = sum(initial_probabilities.values())
                normalized_probabilities = {k: v / total_probability for k, v in initial_probabilities.items()}
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

    # return jsonify({
    #     "best_params": {
    #         "function_set": [
    #             "add",
    #             "sub",
    #             "mul",
    #             "div",
    #             "sqrt",
    #             "log",
    #             "abs",
    #             "neg",
    #             "inv"
    #         ],
    #         "generations": 10,
    #         "max_samples": 0.9606856080159376,
    #         "p_crossover": 0.3090626661973152,
    #         "p_hoist_mutation": 0.19093733380268474,
    #         "p_point_mutation": 0.19093733380268474,
    #         "p_subtree_mutation": 0.3090626661973152,
    #         "parsimony_coefficient": 0.0000803428140079688,
    #         "population_size": 1481,
    #         "random_state": 999
    #     },
    #     "best_score": -2.238423601204719
    # }), 200


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

if __name__ == '__main__':
    app.run(debug=True)
