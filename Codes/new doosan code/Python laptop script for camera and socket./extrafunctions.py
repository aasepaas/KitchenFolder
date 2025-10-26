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
    [1379, 104]
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
    [517.11, -692.31, 21.31]
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


