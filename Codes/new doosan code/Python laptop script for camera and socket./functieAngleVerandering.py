import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Nieuwe camera -> Doosan hoekmetingen ===
camera = np.array([
    25.429106402925697,
    19.33717385283245,
    24.809043838318928,
    11.70807071418443,
    7.771031872104288,
    11.510058662524841,
    34.387104032385594,
    16.58310698957007,
    19.206619492630352,
    8.347611785654578,
    16.92062228584126,
    1.3294171020745011,
    38.9634691270424,
    15.051122413520886
]).reshape(-1, 1)

doosan = np.array([
    165,
    194,
    165,
    194,
    174,
    190,
    210,
    170,
    180,
    189,
    170,
    184,
    190,
    194
])

# === Polynomiale regressie (graad 2) ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(camera)

model = LinearRegression().fit(X_poly, doosan)

# === Functie om camera-angle → doosan-angle te berekenen ===
def cameraAngle_to_doosan(angle):
    angle_poly = poly.transform(np.array([[angle]]))
    return float(model.predict(angle_poly)[0])


