import os
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx

# 文件路径
path = './Desktop/training/T-drive Taxi Trajectories/train'
files = [f for f in os.listdir(path) if f.endswith('.txt')]

# 检查files列表是否为空
print("Files to merge:", files)

if not files:
    print("No .txt files found in the directory.")
else:
    open('./Desktop/training/T-drive Taxi Trajectories/train/merged_file.txt', 'a').close()

    # 保存到merged_file.txt文件中
    with open('./Desktop/training/T-drive Taxi Trajectories/train/merged_file.txt', 'w') as outfile:
        for file in files:
            with open(os.path.join(path, file)) as infile:
                outfile.write(infile.read() + '\n')

    # 读取合并后的文件
    gps_data = pd.read_csv('./Desktop/training/T-drive Taxi Trajectories/train/merged_file.txt', names=['vehicle_id', 'timestamp', 'longitude', 'latitude'])

    # 根据vehicle_id和timestamp排序
    gps_data = gps_data.sort_values(by=['vehicle_id', 'timestamp']).reset_index(drop=True)

    # 统计原始数据量
    raw_length = len(gps_data)  # 原始数据行数
    print("Original data length:", raw_length)

    # 数据去重
    gps_data_1 = gps_data.drop_duplicates().reset_index(drop=True)
    print("Data length after removing duplicates:", len(gps_data_1))  # 去重后的数据行数

    # 筛选北京市五环内的数据 (116.17 - 116.62, 39.83 - 40.05)
    gps_data_2 = gps_data_1[(gps_data_1['latitude'] > 39.83) &
                            (gps_data_1['latitude'] < 40.05) &
                            (gps_data_1['longitude'] > 116.17) &
                            (gps_data_1['longitude'] < 116.62)]
    gps_data_2 = gps_data_2.reset_index(drop=True)

    print("Filtered data length (within 5th Ring Road):", len(gps_data_2))  # 筛选后的数据行数

   
# 获取北京市五环范围内的路网数据
beijing_road = ox.graph_from_bbox(40.05, 39.83, 116.62, 116.17, network_type='drive')

# 保存为shapefile文件
ox.save_graph_shapefile(beijing_road, './Desktop/training/T-drive Taxi Trajectories/train')

# 可视化路网
fig, ax = ox.plot_graph(beijing_road, figsize=(15, 15), show=False, close=False, node_size=4)

# 提取GPS点的经纬度
latitude = gps_data_2.latitude.to_list()
longitude = gps_data_2.longitude.to_list()

# 将经纬度组合成坐标对，用于KMeans聚类
coords = np.array(list(zip(latitude, longitude)))

# 使用KMeans进行聚类，找到前500个最多交汇的路段
kmeans = KMeans(n_clusters=500, random_state=0).fit(coords)

# 获取聚类中心 (交汇的地标)
centers = kmeans.cluster_centers_

# 在图中标出聚类中心，用蓝色 "X" 显示
ax.scatter(centers[:, 1], centers[:, 0], s=100, c='blue', marker='x', label="Top 500 Intersections")

# 显示图例
plt.legend()
plt.show()

# --- 最短路径规划，尽可能经过聚类中心 ---

# 获取北京市五环范围内的路网数据
G = ox.graph_from_bbox(40.05, 39.83, 116.62, 116.17, network_type='drive')

# 定义起点和终点（输入经纬度）
origin_point = (39.90, 116.40)  # 起点的经纬度
destination_point = (39.95, 116.50)  # 终点的经纬度

# 获取起点和终点的最近节点
origin_node = ox.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])  # X 是经度，Y 是纬度
destination_node = ox.nearest_nodes(G, X=destination_point[1], Y=destination_point[0])

# 计算默认的最短路径（使用Dijkstra算法）
shortest_path = nx.shortest_path(G, origin_node, destination_node, weight='length')

# 路径可视化（默认最短路径）
fig, ax = ox.plot_graph_route(G, shortest_path, route_linewidth=6, node_size=0, bgcolor='k')

# 增加经过聚类中心的约束
# 计算到每个聚类中心的最短路径，并找到能够经过最多聚类点的路径
paths_with_clusters = []
for center in centers:
    center_node = ox.nearest_nodes(G, X=center[1], Y=center[0])

    # 起点 -> 聚类中心 -> 终点 的路径
    path_via_center = nx.shortest_path(G, origin_node, center_node, weight='length')
    path_via_center += nx.shortest_path(G, center_node, destination_node, weight='length')[1:]  # 防止重复连接点

    # 保存经过聚类点的路径
    paths_with_clusters.append(path_via_center)

# 选择最优路径（经过聚类中心且长度最短的路径）
best_path = min(paths_with_clusters, key=lambda p: nx.path_weight(G, p, weight='length'))

# 获取最佳路径中的所有节点的经纬度
best_path_latlon = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in best_path]

# 打印经过最佳路径的节点坐标
print("Best Path Nodes (latitude, longitude):")
for lat, lon in best_path_latlon:
    print(f"({lat}, {lon})")

# 可视化最佳路径并标注经过的节点
fig, ax = ox.plot_graph_route(G, best_path, route_linewidth=6, node_size=0, bgcolor='k', route_color='r')

# 在路径上标出每个经过的节点，用红色圆点表示
best_path_nodes_x = [G.nodes[node]['x'] for node in best_path]
best_path_nodes_y = [G.nodes[node]['y'] for node in best_path]
ax.scatter(best_path_nodes_x, best_path_nodes_y, s=100, c='green', zorder=5, label="Best Path Nodes")

plt.legend()
plt.show()
