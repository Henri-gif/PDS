import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Output directory
output_dir = "C:/Users/User/Desktop/PFE/PDS/GitHub/VAE/With synthetic data/output"

# 1. Load and preprocess AIS data from CSV
def load_ais_data(csv_file_path):
    """
    Load AIS data from CSV file with columns: t, AgentID, speed, is_anomaly, anomaly_type, x, y
    """
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Anomaly distribution:\n{df['is_anomaly'].value_counts()}")
    
    # Extract features for the model
    # Using speed, x, y as features 
    feature_columns = ['speed', 'x', 'y']
    
    # Check if all required columns exist
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Extract features and labels
    data = df[feature_columns].values
    labels = df['is_anomaly'].values
    
    return data, labels, df

# 2. VAE Model Definition using TensorFlow/Keras
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(hidden_dim//2, activation="relu"),
        ])
        
        # Latent space means and log variances
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(hidden_dim//2, activation="relu"),
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        # Validation metrics
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]
    
    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        return reconstructed
    
    def train_step(self, data):
        # For autoencoders, we use the same data as input and target
        x, y = data
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(x)
            reconstruction = self.decode(z)
            
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(x, reconstruction)
            )
            reconstruction_loss *= self.input_dim
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        
        z_mean, z_log_var, z = self.encode(x)
        reconstruction = self.decode(z)
        
        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(x, reconstruction)
        )
        reconstruction_loss *= self.input_dim
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        
        total_loss = reconstruction_loss + kl_loss
        
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        
        return {
            "val_loss": self.val_total_loss_tracker.result(),
            "val_reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "val_kl_loss": self.val_kl_loss_tracker.result(),
        }

# 3. Training Function
def train_vae(model, train_data, val_data, epochs=50, lr=1e-3, batch_size=64):
    # Compile model with a dummy loss function (the actual loss is computed in train_step)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_data))
    val_dataset = val_dataset.batch(batch_size)
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=1
    )
    
    return history

# 4. Anomaly Detection Function
def detect_anomalies(model, data, threshold_multiplier=2.0):
    # Get reconstructions
    reconstructions = model.predict(data, verbose=0)
    
    # Calculate reconstruction error for each sample
    reconstruction_errors = np.mean(np.square(data - reconstructions), axis=1)
    
    # Calculate threshold (mean + multiplier * std)
    threshold = np.mean(reconstruction_errors) + threshold_multiplier * np.std(reconstruction_errors)
    
    # Identify anomalies
    anomalies = reconstruction_errors > threshold
    
    return reconstruction_errors, threshold, anomalies, data, reconstructions




# 5. Visualization Functions
def plot_training_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    
    # Check if validation loss exists before plotting
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved as '{output_dir}/training_loss.png'")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Anomaly']):
    """
    Plot a detailed confusion matrix with annotations and interpretations
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix - Anomaly Detection Results', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add interpretation text
    total_samples = np.sum(cm)
    accuracy = np.trace(cm) / total_samples
    
    # Calculate metrics for each class
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{output_dir}/confusion_matrix.png'")
    plt.show()
    plt.close()
    
    return cm, tn, fp, fn, tp, precision, recall, f1

def plot_roc_curve(y_true, reconstruction_errors):
    """
    Plot ROC curve for anomaly detection
    """
    # For ROC curve, we use reconstruction errors as anomaly scores
    # Higher error = more likely to be anomaly
    fpr, tpr, thresholds = roc_curve(y_true, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Find optimal threshold (Youden's J statistic)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot optimal point
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, 
                label=f'Optimal Threshold: {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarm Rate)', fontsize=12)
    plt.ylabel('True Positive Rate (Detection Rate)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    print(f"ROC curve saved as '{output_dir}/roc_curve.png'")
    plt.show()
    plt.close()
    
    return fpr, tpr, roc_auc, optimal_threshold

def plot_precision_recall_curve(y_true, reconstruction_errors):
    """
    Plot Precision-Recall curve for anomaly detection
    """
    precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_errors)
    average_precision = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    
    # Plot random classifier baseline (for imbalanced data)
    random_precision = np.sum(y_true) / len(y_true)
    plt.axhline(y=random_precision, color='red', linestyle='--', 
                label=f'Random Classifier (AP = {random_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # Find optimal threshold (F1-score maximization)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_pr = thresholds[optimal_idx]
    
    # Plot optimal point
    plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='green', s=100,
                label=f'Optimal F1 Threshold: {optimal_threshold_pr:.3f}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved as '{output_dir}/precision_recall_curve.png'")
    plt.show()
    plt.close()
    
    return precision, recall, average_precision, optimal_threshold_pr

def plot_anomalies(losses, threshold, labels):
    plt.figure(figsize=(12, 6))
    
    # Plot reconstruction errors
    plt.subplot(1, 2, 1)
    colors = ['blue' if label == 0 else 'red' for label in labels]
    plt.scatter(range(len(losses)), losses, c=colors, alpha=0.6)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error with Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    
    # Create custom legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Anomaly')
    ]
    plt.legend(handles=legend_elements)
    
    # Plot distribution of errors
    plt.subplot(1, 2, 2)
    sns.histplot(losses[labels == 0], kde=True, color='blue', label='Normal', alpha=0.7)
    sns.histplot(losses[labels == 1], kde=True, color='red', label='Anomaly', alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anomaly_detection.png", dpi=300, bbox_inches='tight')
    print(f"Anomaly detection plot saved as '{output_dir}/anomaly_detection.png'")
    plt.show()
    plt.close()

def plot_reconstructions(original, reconstructed, n_samples=5):
    plt.figure(figsize=(12, 8))
    
    # Select random samples
    indices = np.random.choice(range(len(original)), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(n_samples, 2, 2*i+1)
        plt.bar(range(original.shape[1]), original[idx], alpha=0.7, label='Original')
        plt.title(f'Sample {idx}: Original')
        if i == 0:
            plt.legend()
        
        plt.subplot(n_samples, 2, 2*i+2)
        plt.bar(range(reconstructed.shape[1]), reconstructed[idx], alpha=0.7, color='orange', label='Reconstructed')
        plt.title(f'Sample {idx}: Reconstructed')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reconstructions.png", dpi=300, bbox_inches='tight')
    print(f"Reconstructions plot saved as '{output_dir}/reconstructions.png'")
    plt.show()
    plt.close()

def plot_data_distribution(df):
    """Plot distribution of features and anomaly types"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Speed distribution
    axes[0, 0].hist(df['speed'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Speed Distribution')
    axes[0, 0].set_xlabel('Speed')
    axes[0, 0].set_ylabel('Frequency')
    
    # Position scatter plot
    scatter = axes[0, 1].scatter(df['x'], df['y'], c=df['is_anomaly'], cmap='coolwarm', alpha=0.6)
    axes[0, 1].set_title('Position with Anomalies')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(scatter, ax=axes[0, 1], label='Anomaly')
    
    # Anomaly type distribution
    if 'anomaly_type' in df.columns:
        anomaly_counts = df['anomaly_type'].value_counts()
        axes[1, 0].bar(anomaly_counts.index, anomaly_counts.values)
        axes[1, 0].set_title('Anomaly Type Distribution')
        axes[1, 0].set_xlabel('Anomaly Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Agent distribution (top 10)
    agent_counts = df['AgentID'].value_counts().head(10)
    axes[1, 1].bar(range(len(agent_counts)), agent_counts.values)
    axes[1, 1].set_title('Top 10 Agents by Number of Points')
    axes[1, 1].set_xlabel('Agent Index')
    axes[1, 1].set_ylabel('Number of Points')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Data distribution plot saved as '{output_dir}/data_distribution.png'")
    plt.show()
    plt.close()

# Main execution
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = "C:/Users/User/Desktop/PFE/PDS/GitHub/synthetic_vessel_tracks_with_anomalies_20251007.csv"  # Update this path to your CSV file
    
    print("Loading AIS data from CSV...")
    # Load data from CSV
    data, labels, df = load_ais_data(csv_file_path)
    
    # Plot data distribution
    plot_data_distribution(df)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Filter normal data for training and validation
    train_normal_mask = y_train == 0
    X_train_normal = X_train_scaled[train_normal_mask]
    
    test_normal_mask = y_test == 0
    X_test_normal = X_test_scaled[test_normal_mask]
    
    print("Creating and training VAE model...")
    # Create VAE model
    input_dim = X_train.shape[1]
    vae = VAE(input_dim=input_dim, hidden_dim=64, latent_dim=16)
    
    # Train the model on normal data only
    history = train_vae(
        vae, X_train_normal, X_test_normal, epochs=50, lr=1e-3, batch_size=64
    )
    
    # Plot training loss
    plot_training_loss(history)
    
    print("Detecting anomalies...")
    # Detect anomalies on full test set
    reconstruction_errors, threshold, pred_anomalies, original, reconstructed = detect_anomalies(
        vae, X_test_scaled, threshold_multiplier=2.0
    )
    
    # Calculate metrics
    print("\n" + "="*50)
    print("ANOMALY DETECTION RESULTS")
    print("="*50)
    print(classification_report(y_test, pred_anomalies, target_names=['Normal', 'Anomaly']))
    
    accuracy = accuracy_score(y_test, pred_anomalies)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    cm, tn, fp, fn, tp, precision, recall, f1 = plot_confusion_matrix(y_test, pred_anomalies)
    
    # Plot ROC curve
    print("\nGenerating ROC Curve...")
    fpr, tpr, roc_auc, optimal_threshold_roc = plot_roc_curve(y_test, reconstruction_errors)
    
    # Plot Precision-Recall curve
    print("Generating Precision-Recall Curve...")
    precision_pr, recall_pr, average_precision, optimal_threshold_pr = plot_precision_recall_curve(y_test, reconstruction_errors)
    
    # Plot anomalies
    plot_anomalies(reconstruction_errors, threshold, y_test)
    
    # Plot some reconstructions
    plot_reconstructions(X_test_scaled, reconstructed, n_samples=5)
    
    # Save model and scaler
    vae.save_weights(f"{output_dir}/vae_model.weights.h5")
    print(f"Model weights saved as '{output_dir}/vae_model.weights.h5'")
    
    import joblib
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"Scaler saved as '{output_dir}/scaler.pkl'")
    
    # Save processed data for reference
    processed_data = {
        'data': data,
        'labels': labels,
        'feature_names': ['speed', 'x', 'y']  # Update if you use different features
    }
    np.savez(f"{output_dir}/processed_ais_data.npz", **processed_data)
    print(f"Processed data saved as '{output_dir}/processed_ais_data.npz'")
    
    # Save results
    results = {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'input_dim': input_dim,
        'n_samples': len(data),
        'anomaly_ratio': float(np.mean(labels)),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'roc_auc': float(roc_auc),
        'auprc': float(average_precision),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'optimal_threshold_roc': float(optimal_threshold_roc),
        'optimal_threshold_pr': float(optimal_threshold_pr)
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved as '{output_dir}/results.json'")
    
    print("\n" + "="*50)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Generated files:")
    print(f"- {output_dir}/confusion_matrix.png: Detailed performance analysis")
    print(f"- {output_dir}/roc_curve.png: ROC curve with AUC score")
    print(f"- {output_dir}/precision_recall_curve.png: Precision-Recall curve with AP score")
    print(f"- {output_dir}/training_loss.png: Model training progress")
    print(f"- {output_dir}/anomaly_detection.png: Anomaly detection visualization")
    print(f"- {output_dir}/reconstructions.png: Original vs reconstructed samples")
    print(f"- {output_dir}/data_distribution.png: Data exploration plots")
    print(f"- {output_dir}/vae_model.weights.h5: Trained model weights")
    print(f"- {output_dir}/scaler.pkl: Data preprocessing scaler")
    print(f"- {output_dir}/results.json: Detailed results and metrics")