# soccer-cv
### A python library for soccer (football) data and visualization abstraction, just from video input

<!-- Replace USER, REPO, BRANCH -->
<video controls muted playsinline preload="metadata" width="640" style="max-width:100%;">
  <source src="https://private-user-images.githubusercontent.com/144488510/489017846-ee1854ca-78ae-45c5-ab05-99812ef2860b.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTc3MTEwNjQsIm5iZiI6MTc1NzcxMDc2NCwicGF0aCI6Ii8xNDQ0ODg1MTAvNDg5MDE3ODQ2LWVlMTg1NGNhLTc4YWUtNDVjNS1hYjA1LTk5ODEyZWYyODYwYi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkxMlQyMDU5MjRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01OGViZTI5MGNlOThjN2M5N2MzNTgxMzYzY2Y2ZWNjNmJkOTgwNmJiYTE3YjFhZmNhYmY2NjI4OTY2ZjU0MmI4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.AKSEAOwbfK4_KYbwlfx-PpR1m8DcOtXupMVztSqEa4g" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls muted playsinline preload="metadata" width="640" style="max-width:100%;">
  <source src="https://private-user-images.githubusercontent.com/144488510/489018237-12f8f50f-c19a-4ba3-9e7d-946e569f56de.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTc3MTEzMzEsIm5iZiI6MTc1NzcxMTAzMSwicGF0aCI6Ii8xNDQ0ODg1MTAvNDg5MDE4MjM3LTEyZjhmNTBmLWMxOWEtNGJhMy05ZTdkLTk0NmU1NjlmNTZkZS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkxMlQyMTAzNTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02ZmJiOGQ4ODk4NzliNjRlNDAzMTY2NmM5M2Q4ZjEyNjRhNzQ0ZDhhMGUwMTY3ZjU0ODZkZjY2MDk4ZjY4NzkyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hBOhnt8zTEr0UV1um3EIDq9QX_Qcg2XMQTm5HAG6rt4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls muted playsinline preload="metadata" width="640" style="max-width:100%;">
  <source src="https://private-user-images.githubusercontent.com/144488510/489018065-eee7f3f8-4e7c-4d18-95cb-fac69380efbc.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTc3MTEzMzEsIm5iZiI6MTc1NzcxMTAzMSwicGF0aCI6Ii8xNDQ0ODg1MTAvNDg5MDE4MDY1LWVlZTdmM2Y4LTRlN2MtNGQxOC05NWNiLWZhYzY5MzgwZWZiYy5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkxMlQyMTAzNTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03ZWJiZDk5MzYyMmJmYTQ5NDEyNGVkZTQ4ZGE2MWFhYWZjZmY2ODJjYjE5NjRlOGFiNWY5NmZlZTY3ZmUxN2Y5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.tyH1Xn2OlhZrza3fDHrvQ1H6CyIFMul8iuISjtYpANM" type="video/mp4">
  Your browser does not support the video tag.
</video>


|   team_id_mode |   track_id |   samples |   duration_s |   total_distance_m |   distance_per_min_m |   mean_speed_m_s |   median_speed_m_s |   p95_speed_m_s |   max_speed_m_s |   hi_time_s |   sprint_time_s |   hi_distance_m |   accel_events |   max_accel_mag_m_s2 |   stops |
|---------------:|-----------:|----------:|-------------:|-------------------:|---------------------:|-----------------:|-------------------:|----------------:|----------------:|------------:|----------------:|----------------:|---------------:|---------------------:|--------:|
|              0 |          6 |       718 |        29.96 |              78.78 |               157.77 |             1.94 |               1.7  |            3.85 |            5.31 |        0.6  |            0    |            3.11 |             79 |                27.65 |       4 |
|              0 |          7 |       210 |         8.36 |              37.25 |               267.31 |             3.24 |               3.29 |            5.1  |            5.17 |        1.12 |            0    |            5.7  |            106 |                 5.72 |       0 |
|              0 |          9 |       742 |        29.96 |             107.11 |               214.51 |             2.7  |               2.8  |            4.92 |            5.32 |        1.28 |            0    |            6.67 |            164 |                 5.1  |       1 |
|              0 |         10 |        28 |         1.16 |               4.67 |               241.69 |             2.31 |               2.89 |            3.41 |            3.49 |        0    |            0    |            0    |             18 |                13.07 |       0 |
|              0 |         13 |       154 |         6.12 |              30.7  |               300.99 |             3.74 |               4.02 |            5.3  |            5.34 |        1.04 |            0    |            5.46 |             64 |                 5.94 |       0 |
|              0 |         15 |       750 |        29.96 |              61    |               122.16 |             1.56 |               1.47 |            3.71 |            4.18 |        0    |            0    |            0    |             40 |                 6.55 |       3 |
|              0 |         16 |       746 |        29.96 |              54.41 |               108.96 |             1.41 |               1.22 |            3.26 |            3.49 |        0    |            0    |            0    |             16 |                 7.53 |       5 |
|              0 |         17 |       750 |        29.96 |              83.24 |               166.71 |             2.45 |               1.73 |            6.05 |            6.84 |        2.16 |            0    |           13.55 |             82 |                14.95 |       1 |
|              0 |         19 |       750 |        29.96 |              59.88 |               119.92 |             1.55 |               1.57 |            2.53 |            4.46 |        0    |            0    |            0    |             61 |                 8.28 |       1 |
|              0 |         20 |       699 |        29.96 |             122.53 |               245.39 |             3.57 |               3.53 |            6.23 |            7.57 |        6.52 |            0.48 |           37.37 |            210 |                21.64 |       2 |
|              0 |         24 |         1 |         0    |               0    |                 0    |             0    |               0    |            0    |            0    |        0    |            0    |            0    |              0 |                 0    |       0 |
|              0 |         26 |         5 |         0.16 |               1.15 |               432.16 |             0.27 |               0.4  |            0.51 |            0.53 |        0    |            0    |            0    |              3 |                10.43 |       1 |
|              0 |         28 |       582 |        23.24 |              60.19 |               155.4  |             2.36 |               2.07 |            6.54 |            7.77 |        1.92 |            0.92 |           12.97 |             63 |                15.92 |       2 |
|              0 |         31 |       532 |        21.24 |              43.06 |               121.65 |             1.52 |               1.45 |            2.74 |            3.37 |        0    |            0    |            0    |              6 |                 3.81 |       3 |
|              0 |         32 |       353 |        14.2  |              29.29 |               123.75 |             1.63 |               1.62 |            3.2  |            3.29 |        0    |            0    |            0    |             30 |                 5.65 |       2 |
|              0 |         34 |       374 |        15.2  |              34.51 |               136.21 |             2.06 |               1.45 |            5.9  |            7.44 |        1.28 |            0.28 |            8.08 |             60 |                25.51 |       1 |
|              0 |         35 |       165 |         6.76 |              21.72 |               192.75 |             2.34 |               2.34 |            4.12 |            4.16 |        0    |            0    |            0    |             46 |                14.43 |       1 |
|              0 |         37 |         2 |         0.08 |               0.11 |                81.65 |             0.01 |               0.01 |            0.02 |            0.02 |        0    |            0    |            0    |              0 |                 0.57 |       0 |
|              0 |         38 |        75 |         2.96 |               4.05 |                82.01 |             0.87 |               0.94 |            1.06 |            1.07 |        0    |            0    |            0    |             15 |                 4.47 |       0 |
|              1 |          1 |       750 |        29.96 |              77.92 |               156.06 |             2.19 |               2.12 |            4.46 |            5.18 |        0.72 |            0    |            3.69 |             19 |                 8.24 |       1 |
|              1 |          2 |       664 |        26.64 |              50.72 |               114.23 |             1.33 |               1.08 |            3    |            3.35 |        0    |            0    |            0    |             62 |                 6.8  |       4 |
|              1 |          3 |       750 |        29.96 |              89.03 |               178.29 |             2.35 |               2.27 |            4.37 |            5.23 |        0.68 |            0    |            3.5  |            107 |                 7.47 |       1 |
|              1 |          4 |       370 |        14.76 |              53.58 |               217.82 |             2.7  |               2.79 |            4.3  |            4.67 |        0    |            0    |            0    |             70 |                10.82 |       0 |
|              1 |          5 |       199 |         8.48 |              42.35 |               299.63 |             3.95 |               4.36 |            6.64 |            6.79 |        3.32 |            0    |           20.11 |             75 |                10.06 |       0 |
|              1 |          8 |       743 |        29.96 |              98.31 |               196.89 |             2.72 |               2.97 |            5.24 |            5.82 |        1.88 |            0    |           10.34 |            122 |                 8.58 |       1 |
|              1 |         11 |       744 |        29.96 |              91.86 |               183.96 |             2.29 |               2.07 |            4.28 |            4.74 |        0    |            0    |            0    |             96 |                 4.8  |       3 |
|              1 |         12 |       750 |        29.96 |              67.89 |               135.95 |             1.85 |               1.59 |            3.64 |            3.84 |        0    |            0    |            0    |             26 |                 7.01 |       1 |
|              1 |         14 |       750 |        29.96 |             100.06 |               200.38 |             2.68 |               2.6  |            5.82 |            6.62 |        2.16 |            0    |           13.13 |            115 |                 7.82 |       1 |
|              1 |         18 |       738 |        29.96 |              81.4  |               163.02 |             2.17 |               1.57 |            5.46 |            5.82 |        2.52 |            0    |           13.79 |            146 |                 6.34 |       2 |
|              1 |         21 |       748 |        29.96 |              90.42 |               181.09 |             2.41 |               2.32 |            4.68 |            5.09 |        0.52 |            0    |            2.63 |            102 |                 6.8  |       2 |
|              1 |         27 |         2 |         0.04 |               0.68 |              1020.36 |             0.07 |               0.07 |            0.14 |            0.14 |        0    |            0    |            0    |              1 |                 3.59 |       0 |


# Functionality
### What this library does
Analyst-style visuals and data extraction from ordinary footage. Can be used at any level by anyone. Some examples include:
- Visualize players, ball, refs on a canonical 2d layout. Can overlay with team shape, team control voronoi diagram, posession HUD, etc.
- Player and team heatmaps
- Tracking ball path over time
- Extracting player kinetic data over time such as position, speed, acceleration, etc.

Formats functionality as extensible pipelines. Each function can be imported and run with no needed input besides a video. 

### Pipelines at a glance
```python
# Voronoi control (with continuous control % HUD)
from soccer_cv.pipelines.voronoi2d import write_voronoi_2d_video
write_voronoi_2d_video("content/121364_0.mp4", "output/voronoi_121364_0.mp4")

# Ball path (2D trail over canonical pitch)
from soccer_cv.pipelines.ball_path import write_ball_path_2d_video
write_ball_path_2d_video("content/clip.mp4", "output/ball_path.mp4")

# Rolling possession (nearest-to-ball heuristic, gap handling)
from soccer_cv.pipelines.possession import write_possession_video
write_possession_video("content/clip.mp4", "output/possession.mp4")

# Per-team player heatmap grids (final PNGs; top-10 by data)
from soccer_cv.pipelines.heatmaps import write_team_player_heatmap_grids
write_team_player_heatmap_grids("content/clip.mp4", "output/")

# Team shape (convex hulls + centroid/area/width/depth; cross-faded)
from soccer_cv.pipelines.team_shape import write_team_shape_video
write_team_shape_video("content/clip.mp4", "output/team_shape.mp4")
```

### How it works (brief)
1. Detect players/refs/ball each frame (ball ROI “fast path” where possible).
2. Track non-ball objects (ByteTrack) → stable track_ids.
3. Classify team per track_id with periodic refresh to resist ID switches.
4. Estimate homography from pitch keypoints and smooth it.
5. Project bottom-center anchors to the canonical pitch.
6. Render the chosen visualization, often every k frames + cross-fade to cut jitter.


# Install

## Prereqs
- Python 3.10-3.12
- A clean virtual environment (recommended)
```bash
# pick one
python -m venv .venv && source .venv/bin/activate      # venv
# OR
conda create -n soccer-cv python=3.12 -y && conda activate soccer-cv
```
- Model access
```
# either login (stores a token)
huggingface-cli login
# or set an env var (CI-friendly)
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### GPU CUDA (recommend if available)

1. Preinstall the matching CUDA wheels
``` bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
```

2. Install the soccer_cv library from GitHub
```bash
pip install "git+https://github.com/granthohol/soccer-cv.git@main"
```

### MPS (Apple Silicon)


### CPU

1. Preinstall the CPU PyTorch stack (avoiding large CUDA downloads)
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
```

2. Install the soccer_cv library from GitHub
```bash
pip install "git+https://github.com/granthohol/soccer-cv.git@main"
```



