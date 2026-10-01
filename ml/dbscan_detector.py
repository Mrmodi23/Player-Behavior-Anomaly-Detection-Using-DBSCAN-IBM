"""
ml/dbscan_detector.py
======================
Machine Learning Pipeline for Player Behavior Anomaly Detection using DBSCAN.

Project: Player Behavior Anomaly Detection Using DBSCAN
Problem Statement:
    Use DBSCAN to cluster in-game movement coordinate data and isolate potential
    automated aiming or cheating behavior in multiplayer games.

Key Viva / Academic Concepts:
1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
   An unsupervised density-based algorithm that discovers clusters of arbitrary shapes
   by identifying dense regions separated by sparse areas.
2. Epsilon (eps):
   The radius of the neighborhood around any given data point.
3. min_samples:
   The minimum number of data points required within the eps-neighborhood
   for a point to be considered a 'core point'.
4. Noise / Anomaly Points:
   Points that do not belong to any cluster (neither core nor reachable border points).
   In scikit-learn's DBSCAN, these points are assigned the label -1.
5. Important Clarification:
   DBSCAN does NOT natively generate probability scores (unlike GMM or Logistic Regression).
   Anomaly classification is strictly binary based on density reachability (Label -1 = Anomaly/Noise,
   Labels >= 0 = Clustered/Normal).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional


class MovementDBSCANDetector:
    """
    DBSCAN-based detector for identifying potential anomalous movement patterns
    in multiplayer game telemetry.
    """

    REQUIRED_COLUMNS = ['player_id', 'timestamp', 'x', 'y', 'speed', 'direction']

    def __init__(self, eps: float = 0.5, min_samples: int = 5, feature_cols: Optional[list] = None):
        """
        Initialize detector with DBSCAN hyperparameters.
        
        Args:
            eps: Maximum distance between two samples for neighborhood density.
            min_samples: Minimum samples in an eps-neighborhood to form a core point.
            feature_cols: Features to use for clustering (defaults to x, y, speed, direction).
        """
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.feature_cols = feature_cols or ['x', 'y', 'speed', 'direction']
        self.scaler = StandardScaler()
        self.model: Optional[DBSCAN] = None
        self.last_results: Optional[pd.DataFrame] = None
        self.summary_stats: Dict[str, Any] = {}

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate that the input DataFrame meets format requirements.
        """
        if df is None or df.empty:
            return False, "Dataset is empty."

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"

        # Verify numeric columns can be cast to float
        numeric_cols = ['x', 'y', 'speed', 'direction']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    return False, f"Column '{col}' must contain valid numeric values."

        return True, "Validation successful."

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw dataset:
        1. Make a copy to preserve immutability
        2. Clean numeric types
        3. Handle missing values via median imputation or forward fill
        4. Sort chronologically per player for consistent telemetry analysis
        """
        data = df.copy()

        # Ensure correct datatypes
        data['player_id'] = data['player_id'].astype(str)
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # If any timestamps failed conversion, fill forward or fallback to sequential index
        if data['timestamp'].isna().any():
            data['timestamp'] = data['timestamp'].fillna(pd.Timestamp.now())

        for col in ['x', 'y', 'speed', 'direction']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val if not pd.isna(median_val) else 0.0)

        # Sort chronologically
        data = data.sort_values(by=['player_id', 'timestamp']).reset_index(drop=True)

        # Compute derived kinetic metrics for deeper behavioral analysis
        # delta direction (angular difference 0-180 degrees)
        data['direction_diff'] = 0.0
        for pid, group in data.groupby('player_id'):
            diff = group['direction'].diff().abs()
            # Wrap around 360 degrees
            diff = diff.apply(lambda d: 360 - d if d > 180 else d)
            data.loc[group.index, 'direction_diff'] = diff.fillna(0.0)

        return data

    def fit_detect(self, df: pd.DataFrame, eps: Optional[float] = None, min_samples: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute ML Pipeline:
        Validation -> Imputation -> Feature Scaling -> DBSCAN -> Noise Flagging -> Statistics

        Returns:
            Tuple of (Results DataFrame with labels/status, Summary Statistics Dictionary)
        """
        if eps is not None:
            self.eps = float(eps)
        if min_samples is not None:
            self.min_samples = int(min_samples)

        is_valid, msg = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data Validation Error: {msg}")

        clean_df = self.preprocess(df)

        # Extract features for clustering
        X = clean_df[self.feature_cols].values

        # Scale features so Euclidean distance accounts for scale differences
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit DBSCAN (using single thread to prevent macOS multiprocessing issues)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean', n_jobs=None)
        labels = self.model.fit_predict(X_scaled)

        # Add cluster labels and anomaly status
        clean_df['cluster'] = labels
        # In DBSCAN, cluster -1 represents noise points (isolated outliers)
        clean_df['status'] = np.where(labels == -1, 'ANOMALY', 'NORMAL')
        clean_df['is_anomaly'] = (labels == -1).astype(int)

        self.last_results = clean_df

        # Calculate comprehensive statistics
        total_records = len(clean_df)
        anomalous_records = int((clean_df['cluster'] == -1).sum())
        normal_records = total_records - anomalous_records
        anomaly_percentage = round((anomalous_records / total_records * 100), 2) if total_records > 0 else 0.0

        unique_clusters = set(labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        num_clusters = len(unique_clusters)

        # Cluster distribution breakdown
        cluster_counts = clean_df['cluster'].value_counts().to_dict()
        cluster_distribution = [
            {
                'cluster': int(c),
                'label': f"Cluster {c}" if c != -1 else "Noise / Anomaly",
                'count': int(count),
                'percentage': round(count / total_records * 100, 2)
            }
            for c, count in sorted(cluster_counts.items(), key=lambda item: item[0])
        ]

        # Player-level summaries
        player_summaries = []
        for pid, pgroup in clean_df.groupby('player_id'):
            p_total = len(pgroup)
            p_anomalies = int((pgroup['cluster'] == -1).sum())
            p_pct = round((p_anomalies / p_total * 100), 2) if p_total > 0 else 0.0
            p_avg_speed = round(float(pgroup['speed'].mean()), 2)
            p_max_speed = round(float(pgroup['speed'].max()), 2)
            p_avg_dir_change = round(float(pgroup['direction_diff'].mean()), 2)

            player_summaries.append({
                'player_id': str(pid),
                'total_movements': p_total,
                'normal_movements': p_total - p_anomalies,
                'anomalous_movements': p_anomalies,
                'anomaly_percentage': p_pct,
                'avg_speed': p_avg_speed,
                'max_speed': p_max_speed,
                'avg_direction_change': p_avg_dir_change,
                'suspicion_level': 'HIGH' if p_pct >= 8.0 else ('MEDIUM' if p_pct >= 3.0 else 'LOW')
            })

        # Sort players by anomaly percentage descending
        player_summaries.sort(key=lambda p: p['anomaly_percentage'], reverse=True)

        self.summary_stats = {
            'total_records': total_records,
            'normal_records': normal_records,
            'anomalous_records': anomalous_records,
            'anomaly_percentage': anomaly_percentage,
            'num_clusters': num_clusters,
            'total_players': len(player_summaries),
            'eps': self.eps,
            'min_samples': self.min_samples,
            'cluster_distribution': cluster_distribution,
            'player_summaries': player_summaries,
            'methodology_note': (
                "DBSCAN identifies dense regions of normal player movement and classifies isolated movement "
                "points as potential anomalies. Label -1 indicates density noise. Conventional classification accuracy "
                "cannot be calculated without ground-truth labels."
            )
        }

        return clean_df, self.summary_stats
