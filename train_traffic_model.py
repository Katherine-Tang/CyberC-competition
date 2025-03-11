import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import GCN
from metrics import MAE, MAPE, RMSE
from data_loader import get_loader
from visualize_dataset import show_pred

# 设置随机种子以确保结果可复现
seed = 2020
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 设置图形界面的参数
plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
train_loader, test_loader = get_loader('PEMS04')

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [
  
    GCN(6, 6, 1).to(device),

]

# 训练和评估模型
all_predict_values = []
epochs = 30
for i, model in enumerate(models):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=3e-2)
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_mae, epoch_rmse, epoch_mape = 0.0, 0.0, 0.0, 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:
            data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(device), data['flow_x'].to(device), data['flow_y'].to(device)
            predict_value = model(data)
            loss = criterion(predict_value, data["flow_y"])
            epoch_mae += MAE(data["flow_y"].cpu(), predict_value.cpu())
            epoch_rmse += RMSE(data["flow_y"].cpu(), predict_value.cpu())
            epoch_mape += MAPE(data["flow_y"].cpu(), predict_value.cpu())
            epoch_loss += loss.item()
            num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_mae = epoch_mae / num
        epoch_rmse = epoch_rmse / num
        epoch_mape = epoch_mape / num
        print(f"Model {i+1}: Epoch: {epoch+1}, Loss: {epoch_loss/num:.4f}, MAE: {epoch_mae:.4f}, RMSE: {epoch_rmse:.4f}, MAPE: {epoch_mape:.4f}, Time: {(end_time - start_time)/60:.2f} mins")

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num = 0
        all_predict_value = torch.tensor([])
        all_y_true = torch.tensor([])
        for data in test_loader:
            data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(device), data['flow_x'].to(device), data['flow_y'].to(device)
            predict_value = model(data)
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data["flow_y"]
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()
            num += 1
        epoch_mae = MAE(all_y_true.cpu(), all_predict_value.cpu())
        epoch_rmse = RMSE(all_y_true.cpu(), all_predict_value.cpu())
        epoch_mape = MAPE(all_y_true.cpu(), all_predict_value.cpu())
        print(f"Model {i+1} Test Loss: {total_loss/num:.4f}, MAE: {epoch_mae:.4f}, RMSE: {epoch_rmse:.4f}, MAPE: {epoch_mape:.4f}")

    all_predict_values.append(all_predict_value.cpu())

# 可视化预测结果
show_pred(test_loader, all_y_true.cpu(), all_predict_values)