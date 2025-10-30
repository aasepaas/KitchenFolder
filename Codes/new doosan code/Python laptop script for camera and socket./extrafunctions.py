import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Nieuwe calibratiepunten (camera (pixels) -> robot (mm) X,Y,Z) ===
camera_points = np.array([
    [1349, 153],
    [1099, 491],
    [1691, 242],
    [1571, 621],
    [1066, 342],
    [942, 633],
    [832, 326],
    [1506, 288],
    [1629, 91],
    [1098, 177],
    [795, 126],
    [1668, 175],
    [1776, 631],
    [1379, 104],
    [1358, 475],
    [1443, 318],
    [1504, 544],
    [1412, 336],
    [1534, 238],
    [1386, 317],
    [1487, 546],
    [1592, 641],
    [1391, 530],
    [1436, 224],
    [882, 180],
    [1171, 353],
    [1249, 228],
    [1025, 482],
    [1324, 166],
    [1568, 187],
    [1067, 191],
    [1371, 484],
    [1086, 124],
    [1129  ,490]
], dtype=float)

robot_points = np.array([
    [491.52, -714.79, 21.31],
    [410.70, -828.01, 22.42],
    [601.58, -756.60, 23.6],
    [544.86, -873.39, 24.74],
    [399.34, -781.14, 22.0],
    [357.13, -868.57, 23.0],
    [332.26, -773.41, 21.4],
    [542.95, -765.32, 23.53],
    [595.68, -688.05, 21.4],
    [416.12, -716.01, 21.3],
    [304.56, -689.52, 19.6],
    [604.17, -717.09, 22.87],
    [610.05, -878.7, 27.46],
    [517.11, -692.31, 21.31],
    [494.78, -826.29, 23.8],
    [522.90, -777.54, 23.0],
    [526.09, -854.77, 24.6],
    [512.154, -783.504, 23.189],
    [552.78, -750.36, 22.7],
    [503.24, -777.16, 22.7],
    [524.16, -853.36, 24.7],
    [555.74, -879.72, 25.0],
    [494.04, -848.03, 24.22],
    [522.03, -742.98, 22.3],
    [338.99, -716.21, 25.4],
    [438.95, -783.83, 22.6],
    [462.79, -740.84, 21.5],
    [384.53, -827.91, 22],
    [485.5, -722.3, 20.5],
    [563.95, -732.76, 21],
    [399.59, -725.29, 21.5],
    [479.11, -835.46, 22.7],
    [404.41, -700.38, 19.5],
    [414.59, -834.26, 22.6]
], dtype=float)

# === Maak polynomiale features (2e orde) ===
poly = PolynomialFeatures(degree=2, include_bias=True)
U = poly.fit_transform(camera_points)

# === Fit afzonderlijke lineaire modellen voor X, Y en Z ===
model_x = LinearRegression().fit(U, robot_points[:, 0])
model_y = LinearRegression().fit(U, robot_points[:, 1])
model_z = LinearRegression().fit(U, robot_points[:, 2])

# === Functie om cameracoördinaten → robotcoördinaten (X,Y,Z) te berekenen ===
def camera_to_robot(u, v):
    inp = poly.transform([[u, v]])
    X = float(model_x.predict(inp)[0])
    Y = float(model_y.predict(inp)[0])
    Z = float(model_z.predict(inp)[0])
    return X, Y, Z


