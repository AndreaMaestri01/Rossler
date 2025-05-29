# Rössler System Simulation

[Watch the YouTube video playlist](https://www.youtube.com/playlist?list=PLuLegdoENcSTkLzipMRFx4yOgAMuL36CT)

## Description

This project implements the numerical simulation of the Rössler dynamical system, a well-known example of a three-dimensional chaotic system. The code uses numerical integration methods (4th-order Runge-Kutta and Euler) to solve the system's differential equations and provides graphical visualizations of the temporal evolution, the attractor in three-dimensional space, the bifurcation diagram, and allows the numerical estimation of periodic initial conditions. Furthermore, a module based on recurrent neural networks (LSTM - Long Short-Term Memory) has been integrated to predict the system's temporal evolution from sampled trajectories, enabling the exploration of machine learning techniques in the modeling of chaotic dynamics.

## Repository Structure

```text
rossler-system/
├── rossler/
│   ├── Rossler.py                     # Attractor simulation and 3D/2D plots
│   ├── Rossler_Bifurcation.py         # Bifurcation diagram as a function of parameter c
│   └── Rossler_CI.py                  # Numerical estimation of initial conditions and period
├── Machine_learning/
│   ├── LSTM_Rossler.py                # LSTM model for predicting Rössler system trajectories
│   ├── Improved_LSTM_Rossler.py       # Enhanced LSTM model with Dropout and EarlyStopping
│   └── LSTM_Graph.py                  # Script for visualizing LSTM prediction results
└── README.md                          # Project documentation
```

## Key Features

* **Rossler.py**: Simulation of the Rössler system using RK4 and Euler integration, equilibrium point calculation, and 3D/2D visualizations.
* **Rossler\_Bifurcation.py**: Calculation and visualization of the bifurcation diagram as the parameter $c$ varies.
* **Rossler\_CI.py**: Minimization algorithm (leastsq) to estimate periodic initial conditions and the oscillation period.
* **LSTM\_Rossler.py**: Training an LSTM model (Keras/TensorFlow) to predict Rössler system coordinates from sampled data.
* **Improved\_LSTM\_Rossler.py**: Enhanced LSTM model with Dropout layers and EarlyStopping callbacks, data normalization, and evaluation metrics (MSE and R²).
* **LSTM\_Graph.py**: Graph generation to visualize LSTM model predictions compared to real data.

## Technologies Used

* **Python 3.x**
* **Libraries**:

  * `numpy` (numerical computations)
  * `scipy` (optimization in `Rossler_CI.py`)
  * `matplotlib` (visualizations)
  * `tensorflow` and `keras` (LSTM model building and training)
  * `scikit-learn` (scaler, evaluation metrics)
  * `visualkeras` (LSTM model architecture visualization)

## Usage

### 1. Simulation and plots with `Rossler.py`

Simulates the Rössler attractor with default parameters and displays the 3D plot:

```bash
python rossler/Rossler.py
```

### 2. Bifurcation diagram with `Rossler_Bifurcation.py`

Generates a bifurcation diagram:

```bash
python rossler/Rossler_Bifurcation.py
```

### 3. Estimation of periodic initial conditions with `Rossler_CI.py`

Finds initial conditions and the period of a periodic trajectory:

```bash
python rossler/Rossler_CI.py
```

### 4. Simple LSTM model training with `LSTM_Rossler.py`

Trains an LSTM model on data saved in `RND1.npy` and saves predictions:

```bash
python lstm/LSTM_Rossler.py
```

### 5. Improved LSTM model training with `Improved_LSTM_Rossler.py`

Trains an enhanced model with Dropout and EarlyStopping:

```bash
python lstm/Improved_LSTM_Rossler.py
```

### 6. Visualization of LSTM results with `LSTM_Graph.py`

Generates plots comparing predictions to real data:

```bash
python lstm/LSTM_Graph.py
```

##

