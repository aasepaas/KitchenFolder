import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Nieuwe camera -> Doosan hoekmetingen ===
camera_angles = np.array([
    -34.36251672896199,
    -3.187086627234734,
    -2.8564594921817594,
     3.525580103680752,
   -29.966930861621325,
   -31.904161411594835,
    17.272721623432865,
   -22.381941210417676,
     5.201554169937726,
   -27.371033353169622,
   -14.102928704142878,
    -4.220959880285626,
    11.52143737669236,
    -2.43037417326972,
    -20.864922770354255,
    25.399903984724236,
    17.642708477135493,
    2.9611714308443196,
    5.853614761146057,
    15.956720674999058,############
    7.137985871325603,
    -14.65584958053381,
    20.538841718991932,
    5.177970189853411,
    21.75563228748929,
    -20.976141945644766,
    1.9884957270269399,
    8.755618415043017,
    9.234207408198388,
    25.440865683560716,
    22.043742475083114,
    0.5480779298585254
], dtype=float).reshape(-1, 1)

# Bijbehorende Doosan-waarden (sommige in vierkante haken of losse ints)
doosan_angles = np.array([
    160.0,
    187.0,
    185.0,
    180.0,
    155.0,
    154.0,
    200.0,
    169.0,
    180.0,
    170.0,
    169.0,
    182.0,
    191.0,
    184.0,
    175.0,
    208.0,
    200,
    186.46475134067393,
    188.826,
    197.28499893682618,#################3
    190,
    175,
    201,
    188.3,
    202,
    168.1634481497151,
    189,
    191,
    191.68337771179966,
    205.24545208521937,
    202.34802832852472,
    184.68613449336107
], dtype=float).reshape(-1, 1)

# === Polynomiale regressie (graad 2) ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(camera_angles)

model = LinearRegression().fit(X_poly, doosan_angles)

# === Functie om camera-angle → doosan-angle te berekenen ===
def cameraAngle_to_doosan(angle):
    angle_poly = poly.transform(np.array([[angle]]))
    return float(model.predict(angle_poly)[0])

