from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import copy
import numpy as np
from IPython.display import clear_output
import os


#  = torch.("cuda")

# 读取文件


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


def data_division(uni_X, uni_y):
    X_train, X_test, y_train, y_test = train_test_split(uni_X, uni_y, test_size=0.3)
    return X_train, X_test, y_train, y_test


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
            X, y = X.to(), y.to()  # 确保数据在正确的设备上
            y_pred = model(X)  # 获取模型预测
            predictions.extend(y_pred.cpu().numpy())  # 存储预测值
            actuals.extend(y.cpu().numpy())  # 存储实际值

    return np.array(predictions), np.array(actuals)


X_scaler = StandardScaler()  # StandardScaler() #
y_scaler = StandardScaler()  # StandardScaler() #
cycles = 4  # 需要输入的参数
epoch = 3000  # 需要输入的参数

filename = "test"
data = pd.read_excel(f'{filename}.xlsx').values

feature_add_num = 6  # 需要输入的参数,提醒范围再0-10之间

results = []

X = data[:, :-1]

# 需要输入之前得到的那个文件
data2 = pd.read_csv(f'{filename}_gp_features.csv', encoding='gbk').values

model = ANN(X.shape[1] + feature_add_num).to()

if feature_add_num != 0:
    X = np.concatenate([X, data2[:, 0:feature_add_num]], axis=1)

y = data[:, -1]
y = y.reshape((-1, 1))

uni_X = X_scaler.fit_transform(X)
uni_y = y_scaler.fit_transform(y)

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

    result_dir = f'result/{cycles}次结果'
    os.makedirs(result_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    csv_filename = os.path.join(result_dir, f'{filename}_results_{cycles}.csv')
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
    model_dir = f'result/{cycles}次结果'
    os.makedirs(model_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    torch.save(best_model_state, os.path.join(model_dir, f'{cycles}_{filename}_best_model_{feature_add_num}.pth'))
    print("Best model saved with test loss:", best_loss, "in cycle", best_cycle)
else:
    print("No model improved.")
