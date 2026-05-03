import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

import model.config as config
from model.utils.audio_processing import extract_dual_features
from model.models.hybrid_model import build_fusion_model

def calculate_eer(y_true, y_score):
    """
    Calculates the Equal Error Rate (EER).
    EER is the standard metric for spoofing detection.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def load_data(augment=False):
    X_spec = []
    X_sem = []
    y = []
    
    categories = [(config.FAKE_AUDIO_DIR, 0), (config.REAL_AUDIO_DIR, 1)]
    
    print(f"Extracting Dual-Stream Fusion Features (Augmentation: {augment})...")
    for directory, label in categories:
        if not os.path.exists(directory): continue
        files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        for f in files:
            # Pass augmentation flag
            spec, sem = extract_dual_features(os.path.join(directory, f), augment=augment)
            if spec is not None and sem is not None:
                X_spec.append(spec)
                X_sem.append(sem)
                y.append(label)
                
    return np.array(X_spec), np.array(X_sem), np.array(y)

def train():
    # 1. Load Data
    print("Loading primary research dataset...")
    X_spec, X_sem, y = load_data(augment=False)
    
    if config.USE_AUGMENTATION:
        print("Applying explicit data augmentation (Hard Sample Mining)...")
        X_spec_aug, X_sem_aug, y_aug = load_data(augment=True)
        X_spec = np.concatenate([X_spec, X_spec_aug])
        X_sem = np.concatenate([X_sem, X_sem_aug])
        y = np.concatenate([y, y_aug])

    if len(y) == 0: 
        print("Error: No data found.")
        return

    # Split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=config.VALIDATION_SPLIT, stratify=y)
    
    X_train = [X_spec[train_idx], X_sem[train_idx]]
    X_test = [X_spec[test_idx], X_sem[test_idx]]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 2. Build Fusion Model with Attention & Focal Loss
    model = build_fusion_model(X_spec.shape[1:], X_sem.shape[1:])
    model.summary()
    
    # 3. Train
    print("\nTraining Dual-Stream Fusion Model with Focal Loss...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )
    
    # 4. Final Evaluation (Research Grade)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- RESEARCH EVALUATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Calculate EER
    # y_pred_probs[:, 1] is the score for the 'Real' class
    eer = calculate_eer(y_test, y_pred_probs[:, 1])
    print(f"Equal Error Rate (EER): {eer*100:.4f}%")
    
    # Save training history for Ablation Study
    print(f"\nFusion model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
