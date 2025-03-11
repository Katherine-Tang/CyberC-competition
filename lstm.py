import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # 导入 train_test_split
import matplotlib.pyplot as plt

# 设置数据目录
data_dir = 'T-drive Taxi Trajectories/train'
def load_data(data_dir):
    data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)
            data.append(df[['longitude', 'latitude']].values)
    return np.concatenate(data)

# 创建序列
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :])
        y.append(data[i+sequence_length, :])
    return np.array(X), np.array(y)

# 标准化坐标
scaler = MinMaxScaler(feature_range=(0, 1))
data = load_data(data_dir)
scaled_data = scaler.fit_transform(data)

# 创建序列
sequence_length = 5  # 例如，使用5个坐标点作为输入序列
X, y = create_sequences(scaled_data, sequence_length)

# 划分训练集和测试集
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 调整训练集和测试集的形状
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 2))
testX = testX.reshape((testX.shape[0], testX.shape[1], 2))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 2)))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(trainX, trainY, epochs=20, batch_size=32, verbose=1)

# 预测
testPredict = model.predict(testX)

# 反标准化预测结果
actual = scaler.inverse_transform(testY)
predicted = scaler.inverse_transform(testPredict)

# 计算均方误差和均方根误差
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# 计算每个预测的误差
errors = actual - predicted

# 可视化误差
plt.figure(figsize=(10, 6))

# 绘制实际值和预测值的散点图
plt.subplot(1, 2, 1)
plt.scatter(actual[:, 0], actual[:, 1], label='Actual', alpha=0.5)
plt.scatter(predicted[:, 0], predicted[:, 1], label='Predicted', color='r', alpha=0.5)
plt.title('Actual vs Predicted Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# 绘制误差的直方图
plt.subplot(1, 2, 2)
plt.hist(errors[:, 0], bins=25, alpha=0.5, label='Longitude Error')
plt.hist(errors[:, 1], bins=25, alpha=0.5, label='Latitude Error')
plt.title('Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()

# 在显示图形之前保存图形
plt.savefig('error_visualization.png')  # 保存为PNG文件

# 显示图形
plt.show()