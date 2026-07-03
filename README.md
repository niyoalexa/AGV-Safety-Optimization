# TUAI - Lightweight Model for Collision Avoidance of AGVs in Crowded Environments

The full dataset for this project is here: https://drive.google.com/file/d/1qRU4nztZGTpZCerNhJOYLJNiSKmFF6qm/view?usp=drive_link

## Preprocessing pipeline:
- Reads lidar_data.csv 
- Feature extraction: uses the LiDAR distance columns  as the feature vector for each scan.
- Builds sliding-window sequences (window=5).
- Uses blocked time split + purge gap to reduce leakage.


## Model + Training + Evaluation:
- Model: 2D CNN + GRU. 
- How it works: CNN2D extracts spatial obstacle patterns from each scan-grid, then GRU learns changes across the 5 scans, then a final layer outputs class scores.
- Training: loads X_train_2d, y_train, trains with Adam 
- Evaluation: tests on untouched X_test_2d, y_test
## Modeling Code -> FIS Section
