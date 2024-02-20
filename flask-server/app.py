import os
import tempfile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from gplearn.genetic import SymbolicTransformer

app = Flask(__name__)
app.X = None
CORS(app, supports_credentials=True)
app.secret_key = 'your_secret_key_here'
# 设置文件上传目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 设置临时文件目录
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

uploaded_data = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    _, temp_filepath = tempfile.mkstemp(dir=TEMP_FOLDER)
    file.save(temp_filepath)
    data = pd.read_excel(temp_filepath).values
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
    print(uploaded_data,'dadadad')

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

if __name__ == '__main__':
    app.run(debug=True)
