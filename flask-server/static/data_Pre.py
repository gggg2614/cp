from scipy.io import loadmat
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump,load
from torch import nn

device = torch.device("cuda")

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

X_scaler = load('X_scaler.joblib')
y_scaler = load('y_scaler.joblib')

data_pre = pd.read_excel(f'E:/LBE/预测集合2.xlsx')
data_pre = data_pre.values

X_pre_scaled = X_scaler.transform(data_pre)
input_tensor = torch.from_numpy(X_pre_scaled).float().cuda()

model = ANN(data_pre.shape[1]).cuda()
model.load_state_dict(torch.load(f"E:/LBE/result/1000_noSarOinldata - 副本_best_model_0.pth"))
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
    
predicted_output = prediction.cpu().numpy()
predicted_output_scaled_back = y_scaler.inverse_transform(predicted_output)

xx = pd.DataFrame(predicted_output_scaled_back)
file_path = 'pre_noSarOinl2.xlsx'  # 指定文件路径和名称
xx.to_excel(file_path, index=True) 