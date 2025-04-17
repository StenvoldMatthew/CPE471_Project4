import numpy as np
import pandas as pd
import math
import random

# File name
dataFileName = "EKF_DATA_circle.txt"

# Read CSV file
data = pd.read_csv('../data/' + dataFileName)

# --- Constant variables --- #
dt = 0.001                  # Timestep
linV = 0.14                 # Linear Velocity
wheelBase = 1              # Wheel Base Length

# ---- Covariance Matrices ---- #
Q = np.array([
  [0.0004, 0, 0, 0, 0],
  [0, 0.0004, 0, 0, 0],
  [0, 0, 0.001, 0, 0],
  [0, 0, 0, 0.001, 0],
  [0, 0, 0, 0, 0.001]
])  # Process noise covariance

H = np.eye(5)  # Measurement matrix

R = np.array([
  [0.1, 0, 0, 0, 0],
  [0, 0.1, 0, 0, 0],
  [0, 0, 0.01, 0, 0],
  [0, 0, 0, 0.01, 0],
  [0, 0, 0, 0, 0.01]
])  # Measurement noise covariance

B = np.eye(5)
u = np.array([[0, 0, 0, 0, 0]])

# Initial covariance matrix P
P = np.array([
  [0.01, 0, 0, 0, 0],
  [0, 0.01, 0, 0, 0],
  [0, 0, 0.01, 0, 0],
  [0, 0, 0, 0.01, 0],
  [0, 0, 0, 0, 0.01]
])

# --- Gathered Data Fields --- #
k = data['%time']
Ox = data['field.O_x']
Oy = data['field.O_y']
Ot = data['field.O_t']
It = data['field.I_t']
CoIt = data['field.Co_I_t']
Gx = data['field.G_x']
Gy = data['field.G_y']
CoGx = data['field.Co_gps_x']
CoGy = data['field.Co_gps_y']

# --- Noise injection examples (optional) --- #
# Example for IMU noise
# It = It.to_list()
# CoIt = CoIt.to_list()
# for i in range(1000, 2000):
#   It[i] += random.gauss(0.5, 0.1) - 0.5
#   CoIt[i] += random.gauss(0.5, 0.1) - 0.5

# Example for GPS noise
# Gx = Gx.to_list()
# Gy = Gy.to_list()
# CoGx = CoGx.to_list()
# CoGy = CoGy.to_list()
# for i in range(1000, 2000):
#   Gx[i] += random.gauss(0.5, 0.1) - 0.5
#   Gy[i] += random.gauss(0.5, 0.1) - 0.5
#   CoGx[i] += random.gauss(0.5, 0.1) - 0.5
#   CoGy[i] += random.gauss(0.5, 0.1) - 0.5

# --- Initialization --- #
angV = linV * math.tan(Ot[0]) / wheelBase
X = np.array([[Ox[0], Oy[0], linV, Ot[0], angV]])  # Initial state

print("x_pos,y_pos,theta")
dataString = "{}" + ',' + "{}" + ',' + "{}"

# ---- Data Loop ---- #
for tick in k:
  tick = int(tick)
  angV = linV * math.tan(Ot[tick - 1]) / wheelBase

  A = np.array([
    [1, 0, dt * math.cos(Ot[tick - 1]), 0, 0],
    [0, 1, dt * math.sin(Ot[tick - 1]), 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, dt],
    [0, 0, 0, 0, 1]
  ])

  Z = np.array([[Gx[tick - 1], Gy[tick - 1], linV, It[tick - 1], angV]])

  # ----- Kalman Filter (Placeholder) ----- #
  # Add Kalman filter prediction and update steps here

  # Update measurement covariance R
  R = np.array([
    [CoGx[tick - 1], 0, 0, 0, 0],
    [0, CoGy[tick - 1], 0, 0, 0],
    [0, 0, 0.01, 0, 0],
    [0, 0, 0, CoIt[tick - 1], 0],
    [0, 0, 0, 0, 0.01]
  ])

  angV = linV * math.tan(Ot[tick - 1]) / wheelBase

  # Print filtered output
  print(dataString.format(X[0][0], X[0][1], X[0][3]))

if __name__ == "__main__":
  pass