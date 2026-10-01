"""
app.py
======
Flask Web Application for:
Player Behavior Anomaly Detection Using DBSCAN

Author / Academic Project: Pattern Recognition & Anomaly Detection
Tech Stack: Flask, Pandas, NumPy, scikit-learn, Vanilla JS, CSS3, HTML5, Chart.js
"""

import os
import io
import json
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

from ml.dbscan_detector import MovementDBSCANDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload limit

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, 'player_movements.csv')
RESULTS_DATA_PATH = os.path.join(RESULTS_DIR, 'detected_anomalies.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global in-memory detector and state for fast responsive queries
detector = MovementDBSCANDetector(eps=0.5, min_samples=5)
current_results_df = None
current_stats = None


def initialize_app_data():
    """
    Load sample dataset and run initial DBSCAN detection so the dashboard
    works immediately upon startup without requiring manual upload.
    """
    global current_results_df, current_stats
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            df = pd.read_csv(DEFAULT_DATA_PATH)
            current_results_df, current_stats = detector.fit_detect(df, eps=0.5, min_samples=5)
            # Save baseline results
            current_results_df.to_csv(RESULTS_DATA_PATH, index=False)
            print(f"[STARTUP] Initialized with {len(current_results_df)} records. Found {current_stats['anomalous_records']} anomalies.")
        except Exception as e:
            print(f"[STARTUP WARNING] Failed to initialize default dataset: {e}")
    else:
        print(f"[STARTUP WARNING] Default dataset not found at {DEFAULT_DATA_PATH}")


# Run initialization on module load
initialize_app_data()


# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    """Render the primary single-page analytics application."""
    return render_template('index.html')


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """
    Return comprehensive dashboard telemetry:
    - Overview cards (total players, records, normal, anomalies, clusters, anomaly %)
    - Cluster distribution
    - Time-series activity breakdown
    - 2D coordinate preview samples
    """
    global current_results_df, current_stats
    if current_results_df is None or current_stats is None:
        initialize_app_data()

    if current_results_df is None:
        return jsonify({"success": False, "error": "No data available."}), 404

    # Build time series histogram (group into 15-20 uniform temporal buckets)
    ts_df = current_results_df.copy()
    ts_df['timestamp_dt'] = pd.to_datetime(ts_df['timestamp'])
    ts_df = ts_df.sort_values(by='timestamp_dt')
    
    # Create ~15 bins across the time range
    num_bins = min(15, len(ts_df))
    time_series = []
    if len(ts_df) > 0:
        ts_df['time_bin'] = pd.cut(ts_df['timestamp_dt'], bins=num_bins)
        grouped = ts_df.groupby('time_bin', observed=False)
        for interval, group in grouped:
            label = interval.left.strftime('%H:%M:%S')
            total = len(group)
            anomalies = int((group['cluster'] == -1).sum())
            time_series.append({
                'time_label': label,
                'total': total,
                'normal': total - anomalies,
                'anomalies': anomalies
            })

    # Sample points for the 2D movement map on the dashboard (up to 800 points for crisp visual performance)
    sample_size = min(800, len(current_results_df))
    # Stratified: ensure all anomalies are included so visual anomalies are not lost by random sampling
    anomalies_df = current_results_df[current_results_df['cluster'] == -1]
    normal_df = current_results_df[current_results_df['cluster'] != -1]
    
    anom_sample = anomalies_df
    normal_sample_size = max(0, sample_size - len(anom_sample))
    normal_sample = normal_df.sample(n=min(normal_sample_size, len(normal_df)), random_state=42) if len(normal_df) > 0 else normal_df
    
    combined_sample = pd.concat([normal_sample, anom_sample]).sample(frac=1.0, random_state=42)

    sample_coords = [
        {
            'x': float(row['x']),
            'y': float(row['y']),
            'cluster': int(row['cluster']),
            'status': str(row['status']),
            'player_id': str(row['player_id']),
            'speed': float(row['speed']),
            'direction': float(row['direction'])
        }
        for _, row in combined_sample.iterrows()
    ]

    return jsonify({
        "success": True,
        "stats": {
            "total_players": current_stats['total_players'],
            "total_records": current_stats['total_records'],
            "normal_records": current_stats['normal_records'],
            "anomalous_records": current_stats['anomalous_records'],
            "num_clusters": current_stats['num_clusters'],
            "anomaly_percentage": current_stats['anomaly_percentage'],
            "eps": current_stats['eps'],
            "min_samples": current_stats['min_samples'],
            "explanation": "DBSCAN identifies dense regions of normal player movement and classifies isolated movement points as potential anomalies."
        },
        "cluster_distribution": current_stats['cluster_distribution'],
        "time_series": time_series,
        "sample_coordinates": sample_coords
    })


@app.route('/api/players', methods=['GET'])
def get_players():
    """Return all player summaries sorted by anomaly percentage."""
    global current_stats
    if current_stats is None:
        initialize_app_data()

    if current_stats is None:
        return jsonify({"success": False, "error": "No data available."}), 404

    return jsonify({
        "success": True,
        "players": current_stats['player_summaries']
    })


@app.route('/api/player/<player_id>', methods=['GET'])
def get_player_detail(player_id):
    """
    Return comprehensive trajectory and metrics for a specific player:
    - Overall player profile
    - Full sequential movement trajectory coordinates
    - Flagged anomalous points
    """
    global current_results_df, current_stats
    if current_results_df is None or current_stats is None:
        initialize_app_data()

    p_df = current_results_df[current_results_df['player_id'] == player_id].copy()
    if p_df.empty:
        return jsonify({"success": False, "error": f"Player '{player_id}' not found."}), 404

    p_df = p_df.sort_values(by='timestamp').reset_index(drop=True)

    # Find player summary
    summary = next((p for p in current_stats['player_summaries'] if p['player_id'] == player_id), None)
    if summary is None:
        p_total = len(p_df)
        p_anom = int((p_df['cluster'] == -1).sum())
        summary = {
            'player_id': player_id,
            'total_movements': p_total,
            'normal_movements': p_total - p_anom,
            'anomalous_movements': p_anom,
            'anomaly_percentage': round(p_anom / p_total * 100, 2) if p_total else 0,
            'avg_speed': round(float(p_df['speed'].mean()), 2),
            'max_speed': round(float(p_df['speed'].max()), 2),
            'avg_direction_change': round(float(p_df.get('direction_diff', pd.Series(0)).mean()), 2),
            'suspicion_level': 'HIGH' if (p_anom / p_total * 100) >= 8.0 else 'LOW'
        }

    # Movement trajectory path points
    path_points = [
        {
            'timestamp': str(row['timestamp']),
            'x': float(row['x']),
            'y': float(row['y']),
            'speed': float(row['speed']),
            'direction': float(row['direction']),
            'cluster': int(row['cluster']),
            'status': str(row['status']),
            'is_anomaly': bool(row['cluster'] == -1)
        }
        for _, row in p_df.iterrows()
    ]

    return jsonify({
        "success": True,
        "player": summary,
        "trajectory": path_points
    })


@app.route('/api/detect', methods=['POST'])
def run_detection():
    """
    Run DBSCAN detection pipeline:
    1. Parse eps & min_samples
    2. Read uploaded CSV dataset or fallback to existing dataset
    3. Validate required columns & numerical properties
    4. Handle missing values & scale features
    5. Execute DBSCAN
    6. Save results to results/detected_anomalies.csv
    7. Return updated results and statistics
    """
    global current_results_df, current_stats, detector

    try:
        # Hyperparameters
        eps_val = request.form.get('eps') or (request.json.get('eps') if request.is_json else None) or 0.5
        min_samples_val = request.form.get('min_samples') or (request.json.get('min_samples') if request.is_json else None) or 5

        try:
            eps_val = float(eps_val)
            min_samples_val = int(min_samples_val)
            if eps_val <= 0 or min_samples_val < 1:
                return jsonify({"success": False, "error": "eps must be > 0 and min_samples must be >= 1"}), 400
        except ValueError:
            return jsonify({"success": False, "error": "Invalid numerical values for eps or min_samples"}), 400

        # Check if CSV file was uploaded
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            if not uploaded_file.filename.lower().endswith('.csv'):
                return jsonify({"success": False, "error": "Uploaded file must be a .csv format."}), 400

            try:
                # Read stream without executing
                content = uploaded_file.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(content))
            except Exception as e:
                return jsonify({"success": False, "error": f"Failed to parse uploaded CSV: {str(e)}"}), 400
        else:
            # Fallback to current or default dataset
            if current_results_df is not None:
                df = current_results_df[['player_id', 'timestamp', 'x', 'y', 'speed', 'direction']].copy()
            else:
                df = pd.read_csv(DEFAULT_DATA_PATH)

        # Run detection pipeline
        detector = MovementDBSCANDetector(eps=eps_val, min_samples=min_samples_val)
        results_df, stats = detector.fit_detect(df)

        current_results_df = results_df
        current_stats = stats

        # Persist results
        current_results_df.to_csv(RESULTS_DATA_PATH, index=False)

        return jsonify({
            "success": True,
            "message": "DBSCAN detection completed successfully.",
            "stats": stats
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"An unexpected error occurred during detection: {str(e)}"}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """
    Return paginated, searchable, and sortable movement detection records.
    Query params:
        page (int, default 1)
        page_size (int, default 50, max 200)
        status (string: 'ALL', 'NORMAL', 'ANOMALY')
        player_id (string: optional filter)
        search (string: text search)
        sort_by (string: 'timestamp', 'player_id', 'speed', 'cluster', 'status')
        sort_order ('asc', 'desc')
    """
    global current_results_df, current_stats
    if current_results_df is None or current_stats is None:
        initialize_app_data()

    if current_results_df is None:
        return jsonify({"success": False, "error": "No records found."}), 404

    df = current_results_df.copy()

    # Filtering by Status
    status_filter = request.args.get('status', 'ALL').upper()
    if status_filter in ['NORMAL', 'ANOMALY']:
        df = df[df['status'] == status_filter]

    # Filtering by Player ID
    player_id_filter = request.args.get('player_id', '').strip()
    if player_id_filter and player_id_filter != 'ALL':
        df = df[df['player_id'] == player_id_filter]

    # Search keyword
    search_query = request.args.get('search', '').strip().lower()
    if search_query:
        mask = (
            df['player_id'].str.lower().str.contains(search_query, na=False) |
            df['status'].str.lower().str.contains(search_query, na=False) |
            df['timestamp'].astype(str).str.contains(search_query, na=False)
        )
        df = df[mask]

    # Sorting
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc').lower()
    ascending = (sort_order == 'asc')

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    total_filtered = len(df)

    # Pagination
    try:
        page = max(1, int(request.args.get('page', 1)))
        page_size = min(200, max(10, int(request.args.get('page_size', 50))))
    except ValueError:
        page = 1
        page_size = 50

    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    page = min(page, total_pages)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_records = df.iloc[start_idx:end_idx]

    records_list = [
        {
            "player_id": str(row['player_id']),
            "timestamp": str(row['timestamp']),
            "x": float(row['x']),
            "y": float(row['y']),
            "speed": float(row['speed']),
            "direction": float(row['direction']),
            "cluster": int(row['cluster']),
            "status": str(row['status'])
        }
        for _, row in page_records.iterrows()
    ]

    return jsonify({
        "success": True,
        "records": records_list,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_filtered,
            "total_pages": total_pages
        },
        "summary": {
            "total_records": current_stats['total_records'],
            "normal_records": current_stats['normal_records'],
            "anomalous_records": current_stats['anomalous_records'],
            "num_clusters": current_stats['num_clusters'],
            "anomaly_percentage": current_stats['anomaly_percentage']
        }
    })


@app.route('/api/coordinates', methods=['GET'])
def get_coordinates():
    """
    Return all 2D coordinates categorized by cluster for the interactive map.
    """
    global current_results_df
    if current_results_df is None:
        initialize_app_data()

    if current_results_df is None:
        return jsonify({"success": False, "error": "No coordinates available."}), 404

    # Group points by cluster label for effortless multi-dataset rendering in Chart.js
    cluster_groups = {}
    for _, row in current_results_df.iterrows():
        c = int(row['cluster'])
        if c not in cluster_groups:
            cluster_groups[c] = []
        cluster_groups[c].append({
            'x': float(row['x']),
            'y': float(row['y']),
            'speed': float(row['speed']),
            'direction': float(row['direction']),
            'player_id': str(row['player_id']),
            'timestamp': str(row['timestamp']),
            'status': str(row['status'])
        })

    return jsonify({
        "success": True,
        "clusters": cluster_groups
    })


@app.route('/api/download-results', methods=['GET'])
def download_results():
    """Download detected anomalies and clusters as a CSV file."""
    if not os.path.exists(RESULTS_DATA_PATH):
        if current_results_df is not None:
            current_results_df.to_csv(RESULTS_DATA_PATH, index=False)
        else:
            initialize_app_data()

    if os.path.exists(RESULTS_DATA_PATH):
        return send_file(
            RESULTS_DATA_PATH,
            mimetype='text/csv',
            as_attachment=True,
            download_name='player_movement_dbscan_results.csv'
        )
    return jsonify({"success": False, "error": "Results file not generated yet."}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print(f"Starting Player Anomaly Detection server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
