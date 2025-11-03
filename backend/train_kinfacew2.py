"""
KinFaceW-II Dataset Training Script
====================================
This script trains a deep learning model for kinship verification using the KinFaceW-II dataset.
The dataset contains 4 kinship relations: Father-Son (FS), Father-Daughter (FD), 
Mother-Son (MS), and Mother-Daughter (MD), with 250 pairs each.

Target: 80%+ accuracy using 5-fold cross-validation
"""

import os
import numpy as np
import cv2
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import custom layers
from custom_layers import AbsoluteDifference, L2Distance, CosineSimilarity

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    # Dataset paths
    DATASET_ROOT = 'KinFaceW-II'
    IMAGE_DIR = os.path.join(DATASET_ROOT, 'images')
    META_DIR = os.path.join(DATASET_ROOT, 'meta_data')
    
    # Relationship types
    RELATIONSHIPS = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    RELATIONSHIP_CODES = ['fd', 'fs', 'md', 'ms']
    
    # Image parameters
    IMG_SIZE = 64  # KinFaceW-II images are 64x64
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 150  # Increased for better convergence
    LEARNING_RATE = 0.0005  # Higher initial learning rate
    PATIENCE = 25  # Increased patience for early stopping
    
    # Model save path
    MODEL_DIR = 'model'
    MODEL_NAME = 'kinship_verification_kinfacew2.keras'
    
    # Cross-validation
    N_FOLDS = 5
    
    # Data augmentation
    USE_AUGMENTATION = True

config = Config()

# Create model directory if it doesn't exist
os.makedirs(config.MODEL_DIR, exist_ok=True)


class KinFaceW2DataLoader:
    """Data loader for KinFaceW-II dataset"""
    
    def __init__(self, config):
        self.config = config
        self.data_cache = {}
        
    def load_mat_file(self, mat_path):
        """Load .mat file containing pair information"""
        try:
            mat_data = sio.loadmat(mat_path)
            # The mat file structure: fold, kin/non-kin, image1, image2
            # Extract the main data array (usually stored under a key)
            for key in mat_data.keys():
                if not key.startswith('__'):
                    return mat_data[key]
            return None
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None
    
    def load_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed (should already be 64x64)
            if img.shape[:2] != (self.config.IMG_SIZE, self.config.IMG_SIZE):
                img = cv2.resize(img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def load_relationship_data(self, relationship):
        """Load all data for a specific relationship type"""
        rel_code = self.config.RELATIONSHIP_CODES[self.config.RELATIONSHIPS.index(relationship)]
        mat_file = os.path.join(self.config.META_DIR, f'{rel_code}_pairs.mat')
        
        print(f"\nLoading {relationship} data from {mat_file}...")
        
        # Load mat file
        pairs_data = self.load_mat_file(mat_file)
        if pairs_data is None:
            print(f"Failed to load {mat_file}")
            return None, None, None
        
        print(f"Mat file shape: {pairs_data.shape}")
        
        # Parse pairs data
        # Structure: [fold, kin/non-kin, image1, image2]
        images_1 = []
        images_2 = []
        labels = []
        folds = []
        
        img_dir = os.path.join(self.config.IMAGE_DIR, relationship)
        
        for row in pairs_data:
            # Extract values properly from numpy arrays
            fold = int(row[0].item() if hasattr(row[0], 'item') else row[0])
            label = int(row[1].item() if hasattr(row[1], 'item') else row[1])
            
            # Handle image names - they are stored as strings in the .mat file
            img1_name = row[2][0] if isinstance(row[2], np.ndarray) else str(row[2]).strip()
            img2_name = row[3][0] if isinstance(row[3], np.ndarray) else str(row[3]).strip()
            
            # Load images
            img1_path = os.path.join(img_dir, img1_name)
            img2_path = os.path.join(img_dir, img2_name)
            
            img1 = self.load_image(img1_path)
            img2 = self.load_image(img2_path)
            
            if img1 is not None and img2 is not None:
                images_1.append(img1)
                images_2.append(img2)
                labels.append(label)
                folds.append(fold)
        
        print(f"Loaded {len(labels)} pairs ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
        
        return np.array(images_1), np.array(images_2), np.array(labels), np.array(folds)
    
    def load_all_data(self):
        """Load data from all relationship types"""
        all_images_1 = []
        all_images_2 = []
        all_labels = []
        all_folds = []
        
        for relationship in self.config.RELATIONSHIPS:
            img1, img2, labels, folds = self.load_relationship_data(relationship)
            if img1 is not None:
                all_images_1.append(img1)
                all_images_2.append(img2)
                all_labels.append(labels)
                all_folds.append(folds)
        
        # Concatenate all data
        all_images_1 = np.concatenate(all_images_1, axis=0)
        all_images_2 = np.concatenate(all_images_2, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_folds = np.concatenate(all_folds, axis=0)
        
        print(f"\n{'='*60}")
        print(f"Total dataset size: {len(all_labels)} pairs")
        print(f"Positive pairs (kin): {sum(all_labels)}")
        print(f"Negative pairs (non-kin): {len(all_labels) - sum(all_labels)}")
        print(f"{'='*60}\n")
        
        return all_images_1, all_images_2, all_labels, all_folds


def build_siamese_cnn(input_shape):
    """
    Build a Siamese CNN architecture for kinship verification
    Uses a shared CNN to extract features from both images, then compares them
    """
    
    # Shared feature extractor (CNN)
    def create_base_network(input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 4
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        return models.Model(inputs, x, name='feature_extractor')
    
    # Create the base network
    base_network = create_base_network(input_shape)
    
    # Create two input layers for the two images
    input_1 = layers.Input(shape=input_shape, name='image_1')
    input_2 = layers.Input(shape=input_shape, name='image_2')
    
    # Extract features from both images using the shared network
    features_1 = base_network(input_1)
    features_2 = base_network(input_2)
    
    # Compute multiple distance metrics using custom layers (NO Lambda layers!)
    # 1. Absolute difference - using custom layer
    abs_diff = AbsoluteDifference()([features_1, features_2])
    
    # 2. Element-wise multiplication
    multiply = layers.Multiply()([features_1, features_2])
    
    # 3. Concatenate features
    concat = layers.Concatenate()([features_1, features_2, abs_diff, multiply])
    
    # Classification layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='kinship_prediction')(x)
    
    # Create the full model
    model = models.Model(inputs=[input_1, input_2], outputs=output, name='siamese_kinship_model')
    
    return model


def create_data_generators(config):
    """Create data augmentation generators"""
    if config.USE_AUGMENTATION:
        train_datagen = ImageDataGenerator(
            rotation_range=15,  # Increased rotation
            width_shift_range=0.15,  # Increased shift
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,  # Increased zoom
            brightness_range=[0.85, 1.15],  # Wider brightness range
            fill_mode='nearest',
            shear_range=0.1  # Added shear transformation
        )
    else:
        train_datagen = ImageDataGenerator()
    
    return train_datagen


def train_fold(model, X1_train, X2_train, y_train, X1_val, X2_val, y_val, fold_num, config):
    """Train model on a single fold"""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_num + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")
    print(f"Training samples: {len(y_train)} | Validation samples: {len(y_val)}")
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    checkpoint_path = os.path.join(config.MODEL_DIR, f'best_model_fold_{fold_num+1}.keras')
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        [X1_train, X2_train],
        y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=([X1_val, X2_val], y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    return history, model


def evaluate_model(model, X1_test, X2_test, y_test, fold_num):
    """Evaluate model performance"""
    
    # Predictions
    y_pred_proba = model.predict([X1_test, X2_test], verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_num + 1} Results:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*60}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def plot_training_history(histories, save_path='model/training_history.png'):
    """Plot training history for all folds"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for fold_num, history in enumerate(histories):
            if metric in history.history:
                ax.plot(history.history[metric], label=f'Fold {fold_num+1} Train', alpha=0.6)
                ax.plot(history.history[f'val_{metric}'], label=f'Fold {fold_num+1} Val', alpha=0.6, linestyle='--')
        
        ax.set_title(f'{title} Across Folds')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path='model/confusion_matrices.png'):
    """Plot confusion matrices for all folds"""
    
    n_folds = len(results)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    
    if n_folds == 1:
        axes = [axes]
    
    for fold_num, result in enumerate(results):
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[fold_num],
                   xticklabels=['Non-Kin', 'Kin'],
                   yticklabels=['Non-Kin', 'Kin'])
        axes[fold_num].set_title(f'Fold {fold_num+1}\nAcc: {result["accuracy"]:.4f}')
        axes[fold_num].set_ylabel('True Label')
        axes[fold_num].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def save_results(results, config, save_path='model/training_results.json'):
    """Save training results to JSON"""
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'std_accuracy': np.std([r['accuracy'] for r in results]),
        'std_precision': np.std([r['precision'] for r in results]),
        'std_recall': np.std([r['recall'] for r in results]),
        'std_f1': np.std([r['f1'] for r in results])
    }
    
    # Prepare results for JSON (convert numpy arrays to lists)
    json_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'image_size': config.IMG_SIZE,
            'n_folds': config.N_FOLDS
        },
        'average_metrics': avg_metrics,
        'fold_results': [
            {
                'fold': i+1,
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to {save_path}")
    
    return avg_metrics


def main():
    """Main training function"""
    
    print("\n" + "="*60)
    print("KinFaceW-II Kinship Verification Training")
    print("="*60 + "\n")
    
    # Load data
    print("Step 1: Loading dataset...")
    data_loader = KinFaceW2DataLoader(config)
    X1_all, X2_all, y_all, folds_all = data_loader.load_all_data()
    
    # Prepare for 5-fold cross-validation
    print("\nStep 2: Preparing 5-fold cross-validation...")
    
    histories = []
    results = []
    
    # Train on each fold
    for fold_num in range(config.N_FOLDS):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold_num + 1}/{config.N_FOLDS}")
        print(f"{'#'*60}")
        
        # Split data based on fold
        val_mask = (folds_all == fold_num + 1)
        train_mask = ~val_mask
        
        X1_train, X2_train, y_train = X1_all[train_mask], X2_all[train_mask], y_all[train_mask]
        X1_val, X2_val, y_val = X1_all[val_mask], X2_all[val_mask], y_all[val_mask]
        
        # Build model
        print("\nBuilding model...")
        input_shape = (config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
        model = build_siamese_cnn(input_shape)
        
        if fold_num == 0:
            print("\nModel Architecture:")
            model.summary()
            print(f"\nTotal parameters: {model.count_params():,}")
        
        # Train
        history, trained_model = train_fold(
            model, X1_train, X2_train, y_train,
            X1_val, X2_val, y_val, fold_num, config
        )
        
        histories.append(history)
        
        # Evaluate
        fold_results = evaluate_model(trained_model, X1_val, X2_val, y_val, fold_num)
        results.append(fold_results)
        
        # Clear memory
        del model, trained_model
        tf.keras.backend.clear_session()
    
    # Calculate and display average results
    print("\n" + "="*60)
    print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    avg_metrics = save_results(results, config)
    
    print(f"\nAverage Accuracy:  {avg_metrics['accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f} ({avg_metrics['accuracy']*100:.2f}%)")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"Average Recall:    {avg_metrics['recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    print(f"Average F1-Score:  {avg_metrics['f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_training_history(histories)
    plot_confusion_matrices(results)
    
    # Train final model on all data
    print("\n" + "="*60)
    print("Training final model on all data...")
    print("="*60)
    
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
    final_model = build_siamese_cnn(input_shape)
    
    final_model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for final model
    final_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    final_history = final_model.fit(
        [X1_all, X2_all],
        y_all,
        batch_size=config.BATCH_SIZE,
        epochs=100,  # Increased epochs for better convergence
        validation_split=0.1,
        callbacks=final_callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    final_model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Check if target accuracy achieved
    if avg_metrics['accuracy'] >= 0.80:
        print(f"\n{'='*60}")
        print(f"🎉 SUCCESS! Target accuracy of 80% achieved!")
        print(f"Final accuracy: {avg_metrics['accuracy']*100:.2f}%")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"⚠ Target accuracy of 80% not yet achieved.")
        print(f"Current accuracy: {avg_metrics['accuracy']*100:.2f}%")
        print(f"Consider: increasing epochs, adjusting learning rate, or adding more augmentation")
        print(f"{'='*60}\n")
    
    print("\nTraining complete! 🚀")
    print(f"Model saved to: {final_model_path}")
    print(f"Results saved to: model/training_results.json")
    print(f"Plots saved to: model/training_history.png and model/confusion_matrices.png")


if __name__ == '__main__':
    main()
