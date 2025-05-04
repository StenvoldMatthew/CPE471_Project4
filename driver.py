import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# --- Constants --- #
dt = 0.001
linV = 0.14
wheelBase = 1

def loadData(filepath):
  data = pd.read_csv(filepath)
  return {
    'time': data['%time'],
    'Ox': data['field.O_x'],
    'Oy': data['field.O_y'],
    'Ot': data['field.O_t'],
    'It': data['field.I_t'],
    'CoIt': data['field.Co_I_t'],
    'Gx': data['field.G_x'],
    'Gy': data['field.G_y'],
    'CoGx': data['field.Co_gps_x'],
    'CoGy': data['field.Co_gps_y']
  }

def initializeKalman(initialOt, initialOx, initialOy):
  angV = linV * math.tan(initialOt) / wheelBase
  X = np.array([[initialOx, initialOy, linV, initialOt, angV]])

  P = np.eye(5) * 0.01
  H = np.eye(5)
  B = np.eye(5)
  u = np.zeros((5, 1))

  return X, P, H, B, u

def kalmanStep(X, P, A, Z, Q, R, H, B, u):
  # Prediction
  X = A @ X.T + B @ u
  X = X.T
  P = A @ P @ A.T + Q

  # Update
  S = H @ P @ H.T + R
  K = P @ H.T @ np.linalg.inv(S)
  y = Z.T - H @ X.T
  X = X.T + K @ y
  X = X.T

  I = np.eye(5)
  P = (I - K @ H) @ P

  return X, P

def runKalmanFilter(data, Q):
  X, P, H, B, u = initializeKalman(data['Ot'][0], data['Ox'][0], data['Oy'][0])

  filteredX = []
  filteredY = []
  filteredTheta = []

  for tick in data['time']:
    tick = int(tick)
    angV = linV * math.tan(data['Ot'][tick - 1]) / wheelBase

    A = np.array([
      [1, 0, dt * math.cos(data['Ot'][tick - 1]), 0, 0],
      [0, 1, dt * math.sin(data['Ot'][tick - 1]), 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, dt],
      [0, 0, 0, 0, 1]
    ])

    Z = np.array([[data['Gx'][tick - 1], data['Gy'][tick - 1], linV, data['It'][tick - 1], angV]])

    R = np.array([
      [data['CoGx'][tick - 1], 0, 0, 0, 0],
      [0, data['CoGy'][tick - 1], 0, 0, 0],
      [0, 0, 0.01, 0, 0],
      [0, 0, 0, data['CoIt'][tick - 1], 0],
      [0, 0, 0, 0, 0.01]
    ])

    X, P = kalmanStep(X, P, A, Z, Q, R, H, B, u)

    filteredX.append(X[0][0])
    filteredY.append(X[0][1])
    filteredTheta.append(X[0][3])

  return filteredX, filteredY, filteredTheta, list(data['It']), list(data['Ot'])

def injectNoise(data):
  if applyNoiseGlobally:
    indices = range(len(data['time']))
  else:
    indices = range(noiseRange[0], noiseRange[1])

  # Convert Series to mutable lists
  data['It'] = data['It'].tolist()
  data['CoIt'] = data['CoIt'].tolist()
  data['Gx'] = data['Gx'].tolist()
  data['Gy'] = data['Gy'].tolist()
  data['CoGx'] = data['CoGx'].tolist()
  data['CoGy'] = data['CoGy'].tolist()

  maxNoise = 0.2  # Cap for noise

  for i in indices:
    if addImuNoise:
      noise = random.gauss(0, 0.1)
      data['It'][i] += np.clip(noise, -maxNoise, maxNoise)

    if addImuCovarianceNoise:
      noise = random.gauss(0, 0.1)
      data['CoIt'][i] = addCovNoise(data['CoIt'][i])

    if addGpsPositionNoise:
      noiseX = random.gauss(0, 0.1)
      noiseY = random.gauss(0, 0.1)
      data['Gx'][i] += np.clip(noiseX, -maxNoise, maxNoise)
      data['Gy'][i] += np.clip(noiseY, -maxNoise, maxNoise)

    if addGpsCovarianceNoise:
      data['CoGx'][i] = addCovNoise(data['CoGx'][i])
      data['CoGy'][i] = addCovNoise(data['CoGy'][i])


  return data

def addCovNoise(currentVal, mean=0.0, std=0.1, minVal=0.0001, maxVal=1.0):
    noise = random.gauss(mean, std)
    noisyVal = currentVal + noise
    return np.clip(noisyVal, minVal, maxVal)


def plotResults(filteredX, filteredY, gpsX, gpsY, odoX, odoY,
                filteredTheta, imuHeading, odoHeading, useLines=False):
  plt.figure(figsize=(12, 10))

  # Trajectory plot
  plt.subplot(2, 1, 1)
  if useLines:
    plt.plot(gpsX, gpsY, color='red', label="Raw GPS", alpha=0.5)
    plt.plot(odoX, odoY, color='green', label="Encoder Odometry", alpha=0.5)
    plt.plot(filteredX, filteredY, color='blue', label="Kalman Filter Estimate")
  else:
    plt.scatter(gpsX, gpsY, color='red', label="Raw GPS", alpha=0.5, s=1)
    plt.scatter(odoX, odoY, color='green', label="Encoder Odometry", alpha=0.5, s=1)
    plt.scatter(filteredX, filteredY, color='blue', label="Kalman Filter Estimate", s=1)

  plt.xlabel("X Position")
  plt.ylabel("Y Position")
  plt.title("Trajectory Comparison")
  plt.legend()
  plt.axis("equal")
  plt.grid(True)

  # Heading plot
  plt.subplot(2, 1, 2)
  if useLines:
    plt.plot(range(len(imuHeading)), imuHeading, color='red', label="Raw IMU Heading", alpha=0.5)
    plt.plot(range(len(odoHeading)), odoHeading, color='green', label="Raw Odometry Heading", alpha=0.5)
    plt.plot(range(len(filteredTheta)), filteredTheta, color='blue', label="KF Estimated Heading")
  else:
    plt.scatter(range(len(imuHeading)), imuHeading, color='red', label="Raw IMU Heading", alpha=0.5, s=1)
    plt.scatter(range(len(odoHeading)), odoHeading, color='green', label="Raw Odometry Heading", alpha=0.5, s=1)
    plt.scatter(range(len(filteredTheta)), filteredTheta, color='blue', label="KF Estimated Heading", s=1)

  plt.xlabel("Time Step")
  plt.ylabel("Heading (radians)")
  plt.title("Heading Comparison")
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()


def setNoiseScenario(scenarioId):
  global addImuNoise, addGpsPositionNoise, addImuCovarianceNoise, addGpsCovarianceNoise
  global applyNoiseGlobally, noiseRange

  # Reset all
  addImuNoise = False
  addGpsPositionNoise = False
  addImuCovarianceNoise = False
  addGpsCovarianceNoise = False
  applyNoiseGlobally = True
  noiseRange = (1000, 2000)  # default partial range

  if scenarioId == 0:
    # Baseline (b): No noise
    pass

  elif scenarioId == 1:
    # (c) GPS covariance noise — entire dataset
    addGpsCovarianceNoise = True
    applyNoiseGlobally = True

  elif scenarioId == 2:
    # (c) GPS covariance noise — partial dataset
    addGpsCovarianceNoise = True
    applyNoiseGlobally = False

  elif scenarioId == 3:
    # (d) IMU covariance noise — entire dataset
    addImuCovarianceNoise = True
    applyNoiseGlobally = True

  elif scenarioId == 4:
    # (d) IMU covariance noise — partial dataset
    addImuCovarianceNoise = True
    applyNoiseGlobally = False

  elif scenarioId == 5:
    # (e) GPS position noise + covariance noise — entire dataset
    addGpsPositionNoise = True
    addGpsCovarianceNoise = True
    applyNoiseGlobally = True

  elif scenarioId == 6:
    # (e) GPS position noise + covariance noise — partial dataset
    addGpsPositionNoise = True
    addGpsCovarianceNoise = True
    applyNoiseGlobally = False

  else:
    raise ValueError("Invalid scenarioId. Choose an integer between 0 and 6.")
  
def setQMatrix(testCase=0):
  testingVars = [.0005,.0008,.001,.0012,.0015]
  x = .00008
  y = .00008
  velocity = .005
  thetaNoise = .001
  omegaNoise = 0.001
  Q = np.diag([
      x,               # Noise in x
      y,               # Noise in y
      velocity,        # Noise in linear velocity
      thetaNoise,      # Fixed noise in heading
      omegaNoise       # Fixed noise in angular velocity
  ])

  return Q

# --- Noise Injection Config --- #
addImuNoise = False
addGpsPositionNoise = False
addImuCovarianceNoise = False
addGpsCovarianceNoise = False

# Apply to full dataset or just a portion
applyNoiseGlobally = True
noiseRange = (1000, 2000)  # Only used if applyNoiseGlobally is False



"""
  Set global noise booleans and range settings for report scenarios.

  scenarioId meanings:
  0 - Baseline: No noise
  1 - (c) GPS covariance noise on entire dataset
  2 - (c) GPS covariance noise on partial dataset
  3 - (d) IMU covariance noise on entire dataset
  4 - (d) IMU covariance noise on partial dataset
  5 - (e) GPS position noise with changed covariance (entire dataset)
  6 - (e) GPS position noise with changed covariance (partial dataset)
  """

scenario = 1

if __name__ == "__main__":
  for testCase in range(7):  # Try all combinations
    print("Scenario " + str(testCase))
    Q = setQMatrix()
    setNoiseScenario(testCase)
    dataFilePath = "data/EKF_DATA_circle.txt"
    data = loadData(dataFilePath)

    data = injectNoise(data)  # Apply noise based on config
    filteredX, filteredY, filteredTheta, imuHeading, odoHeading = runKalmanFilter(data, Q)

    # print("x_pos,y_pos,theta")
    # for x, y, t in zip(filteredX, filteredY, filteredTheta):
    #   print(f"{x},{y},{t}")

    plotResults(
      filteredX, filteredY,
      list(data['Gx']), list(data['Gy']),
      list(data['Ox']), list(data['Oy']),
      filteredTheta, imuHeading, odoHeading,
      useLines= False
    )
