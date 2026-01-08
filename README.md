**Smart City Traffic Forecasting & Intelligent Routing System**

This project presents a **data-driven traffic flow forecasting and intelligent routing system** designed for smart city applications. Using large-scale real-world urban traffic datasets, we integrate **data engineering, deep learning, and classical graph algorithms** to support real-time traffic prediction and dynamic route planning in complex urban environments.

We processed and engineered **60,000+ multivariate traffic records**, handling missing data imputation, temporal alignment, normalization, and spatial road-network graph construction using Python (Pandas, NumPy). To model both temporal dynamics and spatial dependencies, we developed a **hybrid LSTM–GCN architecture**, achieving an **8.6% accuracy improvement** over traditional LSTM/GCN baselines based on RMSE, MAE, and MAPE metrics.

In addition, we enhanced **Dijkstra’s algorithm** with **dynamic time-based edge weight adjustments**, enabling adaptive routing under varying traffic conditions and reducing route computation time by **approximately 20%**. The final system delivers a functional prototype capable of forecasting traffic conditions and recommending optimized travel paths, demonstrating strong capability in **spatiotemporal data analysis** and the integration of **deep learning with classical algorithmic techniques**.

**Tech Stack:** Python · Pandas · NumPy · LSTM · GCN · Dijkstra’s Algorithm · LaTeX · Excel

