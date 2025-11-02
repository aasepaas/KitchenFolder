import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Nieuwe calibratiepunten (camera (pixels) -> robot (mm) X,Y,Z) ===
camera_points = np.array([ [1225, 256], [1370, 671], [1072, 473], [885, 224], [748, 538], [1170, 118], [1136, 668], [1303, 406], [804,  395], [1456,  424], [1309  ,590], [1352,  478],
                           [915,  344], [1518  ,246], [1215  ,357], [1153 , 456],
                           [736  ,499],
                           [1374,  351],[1078,  370], [1264, 500], [1221  ,459],
                           [1148 , 473], [1236,  237], [984,  702], [797  ,281],[1018  ,616],
                           [1481  ,183], [1524,  725], [719  ,189], [1512,  433], [1035,  190],[730,  406],[1318  ,617]], dtype=float)

robot_points = np.array([ [448.68, -753.73, 19.49], [487.85, -891.12, 19.4], [402.00, -825.28, 19.5], [338.09, -733.63, 19], [288.47, -850.03, 19],
                          [433.38, -697.83, 19], [420.71, -886.87, 19], [470.51, -807.54, 19], [311.34, -794.99, 19], [519.77, -813.22, 19],
                          [466.09, -869.06, 19], [492.93, -822.47, 19 ], [354.19, -773.49, 19], [554.94, -744.42, 19], [446.43, -781.46, 19],
                          [433.15, -815.65, 19], [294.68, -822.50, 19],
                          [499.79, -780.76, 19.1],[407.72, -786.07, 19], [463.06, -826.86, 19],[453.77, -816.33, 19],
                          [427.75, -821.72, 19], [459.08, -739.62, 19], [373.55, -894.13, 19],
                          [313.06, -751.2, 19], [386.9, -869.02, 19], [543.4, -720.53, 19], [536.8, -904.9, 19],[285.91, -713.55, 19],
                          [543.94, -810.2, 19], [400.36, -716.93, 19],[294.13, -794.58, 19],
                          [477.21, -871.06, 19]], dtype=float)# Fit calibration once at startup

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

