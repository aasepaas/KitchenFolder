import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

camera_points = np.array([
    [1185,477],[1254,290],[1081,415],[934,604],[925,239],[778,383],
    [1369,149],[1336,470],[1031,246],[1310,713],[888,623],[1141,690],
    [1515,674],[1475,386],[1402,177],[1139,153],[818,200],[1235,524],
    [1261,242],[1132,406],[1357,448],[1232,265],[1141,505],[1034,338],
    [1034,338],[1039,309]
], dtype=float)

robot_points = np.array([
    [432.56, -824.19, 19],[460.55, -760.75, 19],[399.08, -802.68, 19],
    [351.79, -867.26, 19],[349.49, -736.07, 19],[297.12, -789.5, 19],
    [503.00, -711.31, 19],[484.02, -822.79, 19],[384.55, -743.71, 19],
    [471.24, -905.12, 19],[342.76, -873.27, 19],[419.44, -895.81, 19],
    [531.91, -891.74, 19],[529.67, -799.97, 19],[510.56, -721.88, 19],
    [423.26, -710.97, 19],[311.51, -726.23, 19.07],[448.73, -837.26, 19],
    [463.46, -743.72, 19],[421.54, -801.52, 19],[491.89, -814.29, 19],
    [453.77, -751.08, 19],[420.78, -832.89, 19],[387.36, -776.19, 19],
    [352.71, -814.03, 19],[386.51, -766.93, 19]
], dtype=float)

# === Maak polynomiale features (2e orde) ===
poly = PolynomialFeatures(degree=2, include_bias=True)
U = poly.fit_transform(camera_points)

# === Fit afzonderlijke lineaire modellen voor X, Y en Z ===
model_x = LinearRegression().fit(U, robot_points[:, 0])
model_y = LinearRegression().fit(U, robot_points[:, 1])
model_z = LinearRegression().fit(U, robot_points[:, 2])

# === Functie om cameracoördinaten → robotcoördinaten (X,Y,Z) te berekenen ===
def camera_to_robotZCoords(u, v):

    inp = poly.transform([[u, v]])
    X = float(model_x.predict(inp)[0])
    Y = float(model_y.predict(inp)[0])
    Z = float(model_z.predict(inp)[0])
    return X, Y, Z

