import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from gplearn.genetic import SymbolicTransformer

filename = "test"
data = pd.read_excel(f'{filename}.xlsx').values

X_scaler = StandardScaler() # StandardScaler() # 
y_scaler = StandardScaler() # StandardScaler() #

X = data[:, :-1]
y = data[:, -1]
y = y.reshape((-1, 1))

uni_X = X_scaler.fit_transform(X)
uni_y = y_scaler.fit_transform(y).ravel()

function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']

# 定义搜索参数的范围
param_range = {
     'population_size': (1000, 10000),
     'generations': (10, 50),
     'parsimony_coefficient': (0.00000000001, 0.0001),
}

# 搜索次数
n_searches = 2 # 需要在页面上输入

best_score = -np.inf
best_params = None

for _ in tqdm(range(n_searches), desc="Searching"):
    
    random_state = randint(1,1000).rvs()
    
    
    params = {
        'population_size': randint(*param_range['population_size']).rvs(random_state=random_state),
        'generations': randint(*param_range['generations']).rvs(random_state=random_state),
        'parsimony_coefficient': uniform(*param_range['parsimony_coefficient']).rvs(random_state=random_state),
        'max_samples' : uniform(0.8, 0.2).rvs(random_state=random_state),
        'function_set': function_set
    }
    
    # 初始分配概率
    initial_probabilities = {
        'p_crossover': uniform(0.1, 0.3).rvs(random_state=random_state),
        'p_subtree_mutation': uniform(0.1, 0.3).rvs(random_state=random_state),
        'p_hoist_mutation': uniform(0.05, 0.2).rvs(random_state=random_state),
        'p_point_mutation': uniform(0.05, 0.2).rvs(random_state=random_state),
    }

    # 归一化概率值
    total_probability = sum(initial_probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in initial_probabilities.items()}

    # 更新参数字典
    params.update(normalized_probabilities)
    
    # 创建模型
    model = SymbolicRegressor(random_state=random_state, **params)
    
    # 评估模型
    score = np.mean(cross_val_score(model, uni_X, uni_y, cv=3))
    
    params['random_state'] = random_state
    # 更新最佳模型
    if score > best_score:
        best_score = score
        best_params = params
    
# 打印最佳参数和分数
print("Best Score:", best_score)
print("Best Parameters:", best_params)

###如果上述结果符合要求再进行下一步，可以考虑加一个按钮点击确认下一步还是重新开始上一步

gp = SymbolicTransformer(
                         generations=best_params['generations'],             # 进化代数
                         population_size=best_params['population_size'],       # 初始化树分支数量
                         hall_of_fame=20,           # 每代中选择几个作为展示
                         n_components=20,            # 最终展示的最优的若干个
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
print(X,y,'xxxxxxyyyyyyyyyy')
print(gp.fit(uni_X, uni_y))
gp_features = gp.transform(X)  # 获得新特征

dataframe = pd.DataFrame({'gpf0':gp_features[:,0],
                          'gpf1':gp_features[:,1],
                          'gpf2':gp_features[:,2],
                          'gpf3':gp_features[:,3],
                          'gpf4':gp_features[:,4],
                          'gpf5':gp_features[:,5],
                          'gpf6':gp_features[:,6],
                          'gpf7':gp_features[:,7],
                          'gpf8':gp_features[:,8],
                          'gpf9':gp_features[:,9],
                         })

dataframe.to_csv(f"{filename}_gp_features.csv", index=False, sep=',')