import numpy as np
import matplotlib.pyplot as plt

from metrics import MAE, MAPE, RMSE

plt.style.use('fivethirtyeight')


def get_flow(filename):
    flow_data = np.load(filename)
    return flow_data['data']

def show_pred(test_loader, all_y_true, all_predict_values):
    node_id = 5
    
    # First day plot
    plt.title(f"Node {node_id} - First day flow")
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    y_true_day = test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                  all_y_true)[:24 * 12, node_id, 0, 0].squeeze()  # Remove extra dimensions if any
    y_pred_day = test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                  all_predict_values[0])[:24 * 12, node_id, 0, 0].squeeze()
    
    plt.plot(y_true_day, label='True')
    plt.plot(y_pred_day, label='GCN pred')
    plt.legend()
    plt.savefig(f"../assets/day_pred_flow_node_{node_id}.png", dpi=400)
    plt.show()
    
    # Two weeks plot
    plt.title(f"Node {node_id} - Two weeks flow")
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    y_true_weeks = test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                    all_y_true)[:, node_id, 0, 0].squeeze()
    y_pred_weeks = test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                    all_predict_values[0])[:, node_id, 0, 0].squeeze()
    
    plt.plot(y_true_weeks, label='True')
    plt.plot(y_pred_weeks, label='GCN pred')
    plt.legend()
    plt.savefig(f"../assets/two_weeks_pred_flow_node_{node_id}.png", dpi=400)
    #plt.show()

    # Metrics evaluation
    mae = MAE(y_true_weeks, y_pred_weeks)
    rmse = RMSE(y_true_weeks, y_pred_weeks)
    mape = MAPE(y_true_weeks, y_pred_weeks)
    
    print(f"GCN Accuracy metrics based on original values: MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")



if __name__ == '__main__':
    traffic_data = get_flow('../dataset/PEMS/PEMS04/data.npz')
    print("data size {}".format(traffic_data.shape))
    # 采样某个结点的数据
    node_id = 224
    plt.plot(traffic_data[: 24 * 12, node_id, 0], label="flow")
    plt.plot(traffic_data[: 24 * 12, node_id, 1], label="speed")
    plt.plot(traffic_data[: 24 * 12, node_id, 2], label="other")
    plt.legend(loc=0)
    plt.savefig("../assets/vis.png")
    plt.show()
