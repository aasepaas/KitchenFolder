import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Nieuwe camera -> Doosan hoekmetingen ===
camera = np.array([
    20.058464737509446,
    4.503627500683268,
    58.842371979744414,
    20.091349003969967,
    -8.374109576206623,
    -24.542553332716533,
    -5.227051760932615,
    7.56943464205404,
    4.079896126935903,
    -8.996446200344328,
    -3.510951288567316,
    23.205179203010967,
    13.379991237027522,
    8.48955381849073,
    -27.025933149152916,
    -29.141298758347602,
    -4.335385753085769,
    50.714160053323695,
    -34.99589880243333,
    -21.928065468095472

]).reshape(-1, 1)

doosan = np.array([
    [205],
    [187],
    [180],
    [190],
    [180],
    [175],
    [180],
    [200],
    [184],
    [178],
    [183],
    [200],
    [193],
    [192],
    [169.7],
    [167],
    [185],
    [160],
    [157],
    [174]
])

# === Polynomiale regressie (graad 2) ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(camera)

model = LinearRegression().fit(X_poly, doosan)

# === Functie om camera-angle → doosan-angle te berekenen ===
def cameraAngle_to_doosan(angle):
    angle_poly = poly.transform(np.array([[angle]]))
    return float(model.predict(angle_poly)[0])


