# Player Behavior Anomaly Detection Using DBSCAN

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.x-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![DBSCAN](https://img.shields.io/badge/Algorithm-DBSCAN%20(Density--Based)-purple.svg)](https://en.wikipedia.org/wiki/DBSCAN)
[![License](https://img.shields.io/badge/Academic-Project-brightgreen.svg)]()

> **Academic Problem Statement:**  
> *"Use DBSCAN to cluster in-game movement coordinate data and isolate potential automated aiming or cheating behavior in multiplayer games."*

---

## 📌 Project Overview

In competitive multiplayer games, malicious gameplay automation—such as aimbot snap-targeting, instant teleportation, speed hacking, and out-of-bounds traversal—violates competitive integrity. Unlike supervised techniques that require labeled cheating datasets, **unsupervised density-based spatial clustering (DBSCAN)** analyzes telemetry coordinates and movement mechanics dynamically.

Legitimate players naturally produce high-density spatial patterns by traversing corridors, objective locations, and common tactical routes at physically bounded movement velocities. In contrast, anomalous automation behaviors manifest as isolated outliers in high-dimensional feature space.

This project delivers a **full-stack pattern recognition and anomaly detection system** combining:
- A scikit-learn DBSCAN machine learning pipeline (`ml/dbscan_detector.py`)
- A modular Python Flask REST API backend (`app.py`)
- An interactive esports analytics dashboard (HTML5, Vanilla CSS3, Vanilla JS, Chart.js)

---

## 🎯 Objectives

1. **Spatial & Kinetic Clustering:** Model in-game spatial coordinates ($x, y$), velocity, and directional angle into standardized feature vectors.
2. **Density-Based Noise Isolation:** Automatically identify anomalous movement points as DBSCAN noise (`cluster = -1`) without requiring pre-labeled training data.
3. **Telemetry Trajectory Visualization:** Present interactive 2D spatial maps and per-player sequential trajectories highlighting suspicious deviations.
4. **Interactive Hyperparameter Tuning:** Enable real-time configuration of Epsilon (`eps`) and Minimum Samples (`min_samples`) with instant visual feedback.
5. **Academic & Viva Rigor:** Explicitly differentiate between density-based noise classification and conventional probability scoring, addressing common evaluation pitfalls in unsupervised learning.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | HTML5, CSS3, Vanilla JavaScript | Responsive esports/SOC dark telemetry dashboard |
| **Data Visualization** | Chart.js 4.x (via CDN) | Real-time scatter, doughnut, bar, and timeline charts |
| **Backend API** | Python Flask 3.x | REST endpoints for dashboard, player queries, CSV downloads |
| **Machine Learning** | scikit-learn (`sklearn.cluster.DBSCAN`) | Unsupervised density-based clustering and outlier isolation |
| **Data Processing** | pandas & NumPy | CSV parsing, missing value imputation, kinematic feature engineering |
| **Preprocessing** | `sklearn.preprocessing.StandardScaler` | Normalizes coordinates and speeds for isotropic Euclidean distance |

*Note: No heavy frontend frameworks (such as React) were used, ensuring beginner-friendly inspection and ease of explanation during college viva demonstrations.*

---

## 📂 Project Structure

```text
ibm/
├── app.py                      # Flask REST application & routing
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive academic documentation
├── generate_dataset.py         # Synthetic telemetry generator script
│
├── data/
│   └── player_movements.csv    # 2,500 sample telemetry records (10 players)
│
├── ml/
│   ├── __init__.py             # Module exporter
│   └── dbscan_detector.py      # Complete DBSCAN pipeline class
│
├── templates/
│   └── index.html              # Modern, responsive single-page dashboard
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark gaming & SOC telemetry styles
│   └── js/
│       └── app.js              # Vanilla JS frontend state & Chart.js logic
│
└── results/
    └── detected_anomalies.csv  # Persisted output with clusters and anomaly flags
```

---

## 📊 Dataset Description & Feature Dictionary

The project includes a synthetic dataset of 2,500 records across 10 distinct players (`Player_Apex_01` to `Player_Cipher_10`) modeling typical multiplayer match telemetry.

### Telemetry Columns:
1. `player_id` *(String)*: Unique identifier for the multiplayer participant.
2. `timestamp` *(DateTime)*: Chronological tick (`YYYY-MM-DD HH:MM:SS.mmm`).
3. `x` *(Float)*: Player X coordinate on the 2D game arena grid (0–1000 units).
4. `y` *(Float)*: Player Y coordinate on the 2D game arena grid (0–1000 units).
5. `speed` *(Float)*: Instantaneous movement velocity in meters per second (m/s). Normal walking/sprinting is bounded between 2.0 and 14.5 m/s.
6. `direction` *(Float)*: Player camera/movement yaw angle in degrees (0.0°–360.0°).

> **Important Terminology Note:**  
> The dataset and dashboard use academic terminology such as **“potential anomalous behavior”** or **“potential automated behavior”**. An anomaly indicates statistical density isolation and does not automatically prove cheating in isolation.

---

## 🧠 DBSCAN Pipeline & Theoretical Explanation

### The Machine Learning Pipeline:
```text
CSV Telemetry Dataset
       ↓
Data Validation (Check required columns & numeric types)
       ↓
Missing Value Imputation & Chronological Ordering
       ↓
Feature Extraction (x, y, speed, direction)
       ↓
Feature Standardization (StandardScaler: Zero Mean, Unit Variance)
       ↓
DBSCAN Density Clustering (eps, min_samples)
       ↓
Noise Identification (Label -1 = Anomaly / Outlier)
       ↓
Kinematic Statistics & Anomaly Percentage Computation
       ↓
Visual Dashboard & Export Generation
```

### Core DBSCAN Concepts:
- **Epsilon ($\epsilon$, `eps`)**: The maximum Euclidean distance between two standardized samples for one to be considered in the neighborhood of the other.
- **Minimum Samples (`min_samples`)**: The minimum number of samples in an $\epsilon$-neighborhood required for a data point to be designated a **Core Point**.
- **Core Point**: A point with at least `min_samples` within its $\epsilon$-neighborhood.
- **Border Point**: A point within the $\epsilon$-neighborhood of a Core Point, but having fewer than `min_samples` neighbors itself.
- **Noise / Anomaly Point**: Any point that is neither a Core Point nor density-reachable from any Core Point. Assigned label `-1`.

### Viva Defense: Why DBSCAN Over K-Means?
1. **Arbitrary Geometry**: Player routes through buildings and choke points form arbitrary curved shapes; K-Means forces spherical clusters.
2. **No Prespecified $K$**: DBSCAN determines the number of clusters automatically based on local point density.
3. **Inherent Outlier Detection**: K-Means forces every outlier into a cluster, whereas DBSCAN explicitly isolates noise as label `-1`.
4. **No False Probability Claim**: DBSCAN does not output a posterior probability distribution. Anomaly classification is binary density isolation.

---

## ⚡ Installation & Setup Instructions

### 1. Clone or Open the Project Directory
Navigate to the project folder:
```bash
cd "/Users/mrkitkat/College/ibm"
```

### 2. Create a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5005
```

*The default sample dataset (`data/player_movements.csv`) is automatically loaded and clustered on startup, displaying live metrics and charts immediately.*

---

## 🖥️ How to Use the Application

### 1. Overview Dashboard
- Review top-level statistics: Total Players, Total Records, Normal Records, Anomalous Records, Number of Clusters, and Anomaly Percentage.
- View the 4 visual telemetry charts: Normal vs. Anomaly Breakdown, Cluster Distribution, Activity Timeline, and 2D Coordinate Scatter Plot.

### 2. Player Analysis Page
- Select any player (e.g. `Player_Shadow_08` or `Player_Apex_01`) from the dropdown.
- Inspect the player's average speed, maximum speed, average angular change, anomaly percentage, and automated **Behavior Suspicion Rating** (`HIGH`, `MEDIUM`, `LOW`).
- View the sequential **Trajectory Path Chart**: lines trace legitimate movement routes, while neon rose markers highlight isolated anomaly ticks.

### 3. 2D Movement Visualization Map
- Explore the interactive arena coordinate map.
- Click legend badges to filter by specific clusters or isolate all anomalous coordinates.
- Click **"Anomalies Only"** to pinpoint isolated spatial leaps across the arena.

### 4. Run DBSCAN / Upload Custom CSV
- Upload any compatible `.csv` file via drag-and-drop.
- Adjust hyperparameters:
  - `eps` (default: `0.5`, step: `0.05`)
  - `min_samples` (default: `5`, step: `1`)
- Click **“RUN DBSCAN DETECTION”** to trigger the animated 10-step ML pipeline and refresh all dashboard analytics automatically.

### 5. Results & CSV Export
- Search, filter by status (`NORMAL` / `ANOMALY`), or filter by specific Player ID.
- Sort table columns by clicking headers.
- Click **“Download Results as CSV”** to export the labeled telemetry output.

---

## 📈 Evaluation & Viva Presentation Guide

### Model Evaluation Metric
When ground-truth labels are absent, conventional supervised metrics (Accuracy, Precision, Recall, F1-Score) cannot be computed. In your viva presentation, state:

> *“Because the synthetic dataset does not contain verified ground-truth cheating labels, conventional classification accuracy cannot be reliably calculated. Evaluation is therefore based on clustering structure and detected noise/anomaly patterns.”*

### Key Evaluation Attributes Displayed:
- Total Clusters Discovered
- Absolute Noise Point Count
- Noise-to-Total Ratio (Global Anomaly Percentage)
- Distribution of records per cluster

---

## ⚠️ Limitations & Future Work

1. **Temporal Clustering**: Current DBSCAN clusters on spatial ($x, y$) and kinematic (speed, direction) dimensions. Future iterations could incorporate $k$-d spatio-temporal trees ($x, y, t$).
2. **Adaptive Density**: DBSCAN assumes a relatively uniform global density threshold $\epsilon$. Advanced algorithms like **OPTICS** or **HDBSCAN** could handle variable-density game areas (e.g., dense spawn rooms vs. sparse open fields).
3. **Client-Side vs. Server-Side Verification**: In production anti-cheat systems, DBSCAN telemetry isolation is combined with server tick deterministic simulation and hardware attestation.
