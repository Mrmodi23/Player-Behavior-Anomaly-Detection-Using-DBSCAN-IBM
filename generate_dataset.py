"""
generate_dataset.py
===================
Generates realistic multiplayer game movement telemetry data for:
'Player Behavior Anomaly Detection Using DBSCAN'

Generates ~2,500 records across 10 distinct players:
- Normal players navigating standard tactical map zones (Bombsite A, Bombsite B, Courtyard, Chokepoints)
- Natural movement speeds (walk: 3-6 m/s, sprint: 8-12 m/s) with gradual directional turns
- Several players displaying occasional or repeated potential anomalous movement behavior:
  * Speed hacking / teleportation spikes (speed 35-75 m/s)
  * Extreme instant angle snapping / spinbot jitter (unnatural directional variance)
  * Out-of-bounds / off-mesh coordinate jumps
- Stored as clean telemetry (player_id, timestamp, x, y, speed, direction)
  without cheating labels, preserving realistic raw data.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_telemetry_dataset(output_path="data/player_movements.csv", num_records=2500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    players = [
        {"id": "Player_Apex_01", "anomaly_tendency": "low"},
        {"id": "Player_Viper_02", "anomaly_tendency": "medium_speed"},     # occasional speed spikes
        {"id": "Player_Ghost_03", "anomaly_tendency": "low"},
        {"id": "Player_Titan_04", "anomaly_tendency": "high_aimbot"},      # sharp angle snapping & jitter
        {"id": "Player_Nova_05", "anomaly_tendency": "low"},
        {"id": "Player_Phoenix_06", "anomaly_tendency": "medium_teleport"},# isolated spatial leaps
        {"id": "Player_Spectre_07", "anomaly_tendency": "low"},
        {"id": "Player_Shadow_08", "anomaly_tendency": "high_composite"},  # erratic speed & snap aim
        {"id": "Player_Echo_09", "anomaly_tendency": "low"},
        {"id": "Player_Cipher_10", "anomaly_tendency": "low"}
    ]

    # Map zones representing dense tactical points of interest (Clusters)
    zones = [
        {"name": "Site_A", "center": (220, 280), "spread": 28},
        {"name": "Site_B", "center": (780, 720), "spread": 32},
        {"name": "Mid_Courtyard", "center": (500, 500), "spread": 35},
        {"name": "Spawn_Alpha", "center": (150, 850), "spread": 20},
        {"name": "Spawn_Bravo", "center": (850, 150), "spread": 22},
        {"name": "Sniper_Choke", "center": (480, 240), "spread": 18},
        {"name": "Underpass", "center": (520, 760), "spread": 25}
    ]

    base_time = datetime(2026, 9, 29, 14, 0, 0)
    records = []

    records_per_player = num_records // len(players)

    for p_info in players:
        pid = p_info["id"]
        tendency = p_info["anomaly_tendency"]

        # Pick initial starting zone
        current_zone = random.choice(zones)
        curr_x = current_zone["center"][0] + np.random.normal(0, current_zone["spread"])
        curr_y = current_zone["center"][1] + np.random.normal(0, current_zone["spread"])
        curr_dir = random.uniform(0, 360)
        curr_speed = random.uniform(4.0, 9.0)
        curr_time = base_time + timedelta(seconds=random.randint(0, 60))

        # Target next zone for patrol / movement path
        target_zone = random.choice(zones)

        for step in range(records_per_player):
            curr_time += timedelta(milliseconds=random.randint(450, 650))

            # Decide if this step exhibits potential anomalous behavior based on profile
            is_anomaly_step = False
            anomaly_type = None

            if tendency == "medium_speed" and random.random() < 0.08:
                is_anomaly_step = True
                anomaly_type = "speed_spike"
            elif tendency == "high_aimbot" and random.random() < 0.16:
                is_anomaly_step = True
                anomaly_type = "aim_snap"
            elif tendency == "medium_teleport" and random.random() < 0.07:
                is_anomaly_step = True
                anomaly_type = "teleport"
            elif tendency == "high_composite" and random.random() < 0.18:
                is_anomaly_step = True
                anomaly_type = random.choice(["speed_spike", "aim_snap", "teleport", "out_of_bounds"])
            elif tendency == "low" and random.random() < 0.012:
                # Slight natural noise / lag spike in normal players
                is_anomaly_step = True
                anomaly_type = random.choice(["micro_jitter", "brief_spike"])

            if is_anomaly_step:
                if anomaly_type == "speed_spike":
                    # Unnatural burst speed (35 - 75 m/s)
                    step_speed = round(random.uniform(36.0, 72.0), 2)
                    step_dir = (curr_dir + random.uniform(-15, 15)) % 360
                    # Jump forward fast
                    rad = np.radians(step_dir)
                    curr_x += np.cos(rad) * (step_speed * 1.5)
                    curr_y += np.sin(rad) * (step_speed * 1.5)

                elif anomaly_type == "aim_snap":
                    # Instant 180° snap or extreme micro-jitter at locked speed
                    step_dir = (curr_dir + random.choice([120, 180, 240]) + random.uniform(-5, 5)) % 360
                    step_speed = round(random.uniform(18.0, 28.0), 2)
                    rad = np.radians(step_dir)
                    curr_x += np.cos(rad) * 8.0
                    curr_y += np.sin(rad) * 8.0

                elif anomaly_type == "teleport":
                    # Discontinuous coordinate shift across the map
                    curr_x = float(np.clip(curr_x + random.choice([-1, 1]) * random.uniform(250, 450), 50, 950))
                    curr_y = float(np.clip(curr_y + random.choice([-1, 1]) * random.uniform(250, 450), 50, 950))
                    step_speed = round(random.uniform(48.0, 85.0), 2)
                    step_dir = random.uniform(0, 360)

                elif anomaly_type == "out_of_bounds":
                    # Coordinate pushed beyond tactical boundary or isolated corner
                    curr_x = float(random.choice([random.uniform(5, 30), random.uniform(960, 995)]))
                    curr_y = float(random.choice([random.uniform(5, 30), random.uniform(960, 995)]))
                    step_speed = round(random.uniform(20.0, 40.0), 2)
                    step_dir = random.uniform(0, 360)

                else: # micro_jitter / brief_spike
                    step_speed = round(random.uniform(16.0, 22.0), 2)
                    step_dir = (curr_dir + random.uniform(70, 110)) % 360
                    rad = np.radians(step_dir)
                    curr_x += np.cos(rad) * 5.0
                    curr_y += np.sin(rad) * 5.0

            else:
                # Normal gameplay movement: navigate towards target zone
                tx, ty = target_zone["center"]
                dist = np.hypot(tx - curr_x, ty - curr_y)

                if dist < 40 or random.random() < 0.04:
                    target_zone = random.choice([z for z in zones if z != target_zone])
                    tx, ty = target_zone["center"]

                # Angle towards destination with slight human hand jitter
                target_dir = np.degrees(np.arctan2(ty - curr_y, tx - curr_x)) % 360
                angle_diff = (target_dir - curr_dir + 180) % 360 - 180
                curr_dir = (curr_dir + np.clip(angle_diff * 0.25, -25, 25) + np.random.normal(0, 3.5)) % 360

                # Human speed variations: crouching, walking, sprinting (4 to 12 m/s)
                mode = random.choices(["walk", "run", "sprint"], weights=[0.25, 0.55, 0.20])[0]
                if mode == "walk":
                    step_speed = float(np.random.normal(4.5, 0.6))
                elif mode == "run":
                    step_speed = float(np.random.normal(8.0, 0.8))
                else:
                    step_speed = float(np.random.normal(11.5, 0.7))

                step_speed = max(1.5, min(step_speed, 14.5))

                rad = np.radians(curr_dir)
                step_dist = step_speed * 0.6
                curr_x += np.cos(rad) * step_dist
                curr_y += np.sin(rad) * step_dist

                # Keep bounded in playable area
                curr_x = float(np.clip(curr_x, 60, 940))
                curr_y = float(np.clip(curr_y, 60, 940))

            records.append({
                "player_id": pid,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "x": round(float(curr_x), 2),
                "y": round(float(curr_y), 2),
                "speed": round(float(step_speed), 2),
                "direction": round(float(curr_dir % 360), 2)
            })

    df = pd.DataFrame(records)
    # Shuffle slightly so players are mixed chronologically like a live telemetry stream
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} telemetry records in '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_telemetry_dataset()
