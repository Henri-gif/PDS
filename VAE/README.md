# VAE-based Anomaly Detection in AIS Data

## Overview

This project implements a Variational Autoencoder (VAE) for anomaly detection in Automatic Identification System (AIS) data for vessel tracks. The model is trained on normal (non-anomalous) data to learn a compressed representation of typical vessel behavior (using features like speed, x-position, and y-position). Anomalies are detected based on high reconstruction errors from the VAE.

The script:
- Loads AIS data from a CSV file.
- Preprocesses and normalizes the data.
- Trains the VAE model using TensorFlow/Keras.
- Evaluates anomaly detection on test data.
- Generates performance metrics (accuracy, precision, recall, F1-score, ROC-AUC, AUPRC).
- Produces visualizations (plots for training loss, confusion matrix, ROC/PR curves, anomaly distributions, etc.).
- Saves the trained model, scaler, processed data, and results to an output directory.

This approach is useful for identifying unusual vessel movements, such as deviations in speed or position, which could indicate anomalies like route deviations or sensor errors.

## Requirements

The script requires Python 3.x and the following libraries:

- `numpy` (>=1.19)
- `pandas` (>=1.0)
- `tensorflow` (>=2.0)  # Includes Keras
- `scikit-learn` (>=0.24)  # For preprocessing, splitting, and metrics
- `matplotlib` (>=3.3)  # For plotting
- `seaborn` (>=0.11)  # For enhanced visualizations
- `joblib`  # For saving the scaler (installed via scikit-learn)
- `json` and `warnings`  # Standard Python libraries

Install dependencies using pip:

```
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

Note: TensorFlow should detect and use GPU if available for faster training. The script prints TensorFlow version and GPU availability at startup.

## Usage

1. **Prepare your data**: Ensure your CSV file has the following columns:
   - `t`: Timestamp (not used in modeling).
   - `AgentID`: Vessel identifier (used for visualization).
   - `speed`: Vessel speed.
   - `is_anomaly`: Binary label (0 for normal, 1 for anomaly).
   - `anomaly_type`: Type of anomaly (optional, used for visualization).
   - `x`: X-position.
   - `y`: Y-position.

   Update the `csv_file_path` variable in the script to point to your CSV file.

2. **Set output directory**: Update `output_dir` in the script to your desired output path (e.g., a folder for saving plots and results).

3. **Run the script**:
   ```
   python vae_anomaly_detection.py
   ```

   - The script will load data, train the VAE on normal samples, detect anomalies on the test set, and generate outputs.
   - Training parameters (e.g., epochs=50, batch_size=64, learning_rate=1e-3) can be adjusted in the `train_vae` function.
   - Anomaly detection threshold can be tuned via `threshold_multiplier` in `detect_anomalies` (default: 2.0, i.e., mean + 2*std of reconstruction errors).

4. **Reproducibility**: Random seeds are set for NumPy and TensorFlow to ensure consistent results.

## Key Components

### Data Loading and Preprocessing
- Loads CSV data into a Pandas DataFrame.
- Extracts features: `speed`, `x`, `y`.
- Splits data into train/test (70/30) with stratification on labels.
- Normalizes using `StandardScaler`.
- Trains only on normal data (`is_anomaly == 0`).

### VAE Model
- Encoder: Two dense layers (ReLU activation) + latent space (mean, log-var, sampling).
- Decoder: Two dense layers (ReLU) + output layer (linear).
- Loss: Reconstruction (MSE) + KL divergence.
- Custom `train_step` and `test_step` for VAE-specific losses.
- Hyperparameters: `hidden_dim=64`, `latent_dim=16` (adjustable in `VAE` init).

### Anomaly Detection
- Computes reconstruction errors on test data.
- Threshold: Mean + multiplier * std of errors.
- Flags samples above threshold as anomalies.

### Evaluation and Visualizations
- Metrics: Classification report, accuracy, confusion matrix.
- Curves: ROC (with optimal threshold via Youden's J), Precision-Recall (with optimal F1 threshold).
- Plots:
  - Training/validation loss over epochs.
  - Confusion matrix heatmap.
  - Reconstruction errors scatter and distribution.
  - Original vs. reconstructed samples (bar plots).
  - Data distributions (speed histogram, position scatter, anomaly types, top agents).

All plots are saved as PNG files in the output directory.

## Output Files

The script generates the following in `output_dir`:
- **Plots**:
  - `training_loss.png`: Loss curves.
  - `confusion_matrix.png`: Annotated confusion matrix.
  - `roc_curve.png`: ROC curve with AUC and optimal threshold.
  - `precision_recall_curve.png`: PR curve with AP and optimal F1 threshold.
  - `anomaly_detection.png`: Error scatter and histograms.
  - `reconstructions.png`: Sample reconstructions.
  - `data_distribution.png`: Feature and label distributions.
- **Model and Data**:
  - `vae_model.weights.h5`: Trained VAE weights.
  - `scaler.pkl`: Fitted StandardScaler.
  - `processed_ais_data.npz`: NumPy archive of data, labels, and feature names.
- **Results**:
  - `results.json`: JSON with metrics (threshold, accuracy, confusion matrix, ROC-AUC, AUPRC, etc.).

## Limitations and Improvements
- Assumes anomalies are rare; model is trained only on normal data (unsupervised anomaly detection).
- Features are limited to speed, x, y; add more (e.g., course, acceleration) for better performance.
- For large datasets, increase batch size or use GPU.
- Hyperparameter tuning (e.g., latent dim, epochs) could improve results.
- No handling for time-series dependencies; consider LSTM-VAE for sequential data.

## License
This project is open-source under the MIT License. Feel free to use and modify.
