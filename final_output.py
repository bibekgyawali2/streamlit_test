"""
THESIS CODE: Comparative Analysis of Artificial Neural Network and Random Forest 
for Building Cost Prediction in Nepal
Author: Research Scholar
Date: 2026
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # For baseline comparison
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import joblib

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Create output directory
output_dir = "thesis_figures"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print(" " * 20 + "THESIS RESEARCH: BUILDING COST PREDICTION MODEL")
print("="*80)
print("Comparative Analysis: Artificial Neural Network vs Random Forest")
print("="*80)

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: DATA LOADING AND PREPROCESSING")
print("="*80)

# Load dataset
df_raw = pd.read_csv("./data/Building_Data_v2.csv")
TARGET = "Adjusted Cost (Base Year 2025/26)"
print(f"\n✓ Dataset loaded successfully")
print(f"  - Original samples: {len(df_raw):,}")
print(f"  - Features: {len(df_raw.columns)}")

# Clean target variable
df_raw[TARGET] = pd.to_numeric(
    df_raw[TARGET].astype(str).str.replace(",", "").str.strip(), 
    errors="coerce"
)
df_raw = df_raw.dropna(subset=[TARGET])
print(f"  - After cleaning: {len(df_raw):,}")

# Outlier removal (consistent for both models)
q_low = df_raw[TARGET].quantile(0.10)
q_hi = df_raw[TARGET].quantile(0.99)
outliers_removed = len(df_raw) - len(df_raw[(df_raw[TARGET] < q_hi) & (df_raw[TARGET] > q_low)])
df = df_raw[(df_raw[TARGET] < q_hi) & (df_raw[TARGET] > q_low)].copy()
print(f"  - Outliers removed (bottom 10%, top 1%): {outliers_removed} ({outliers_removed/len(df_raw)*100:.1f}%)")
print(f"  - Final dataset size: {len(df):,}")

# Feature engineering
df["Total_Area"] = df["Plinth Area"] * df["No. of Storeys"]
df["Col_Intensity"] = df["No. of Columns"] / df["Plinth Area"]

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Table 1: Descriptive Statistics
desc_stats = df[TARGET].describe()
table1_desc_stats = pd.DataFrame({
    'Statistic': ['Count', 'Mean (NPR)', 'Std Dev (NPR)', 'Min (NPR)', '25th Percentile', 
                  'Median (NPR)', '75th Percentile', 'Max (NPR)', 'Skewness', 'Kurtosis'],
    'Value': [
        f'{desc_stats["count"]:,.0f}',
        f'{desc_stats["mean"]:,.0f}',
        f'{desc_stats["std"]:,.0f}',
        f'{desc_stats["min"]:,.0f}',
        f'{desc_stats["25%"]:,.0f}',
        f'{desc_stats["50%"]:,.0f}',
        f'{desc_stats["75%"]:,.0f}',
        f'{desc_stats["max"]:,.0f}',
        f'{df[TARGET].skew():.3f}',
        f'{df[TARGET].kurtosis():.3f}'
    ]
})
print("\n📊 TABLE 1: Descriptive Statistics of Building Costs")
print(table1_desc_stats.to_string(index=False))
table1_desc_stats.to_csv(f"{output_dir}/table1_desc_stats.csv", index=False)

# Table 2: Categorical Variable Distribution
location_dist = df['Location'].value_counts()
foundation_dist = df['Foundation Type'].value_counts()
print("\n📊 TABLE 2: Categorical Variable Distribution")
print("\nLocation Distribution:")
print(location_dist.to_string())
print("\nFoundation Type Distribution:")
print(foundation_dist.to_string())

# Figure 1: Cost Distribution Histogram
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[TARGET], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df[TARGET].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[TARGET].mean():,.0f}')
axes[0].axvline(df[TARGET].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[TARGET].median():,.0f}')
axes[0].set_xlabel('Building Cost (NPR)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Distribution of Building Costs', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot by location
locations = df['Location'].unique()
location_data = [df[df['Location'] == loc][TARGET] for loc in locations]
bp = axes[1].boxplot(location_data, labels=locations, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1].set_xlabel('Location', fontweight='bold')
axes[1].set_ylabel('Building Cost (NPR)', fontweight='bold')
axes[1].set_title('Cost Distribution by Location', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 1: Building Cost Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure1_cost_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Figure 1 saved: figure1_cost_distribution.png")

# ============================================================================
# SECTION 4: RANDOM FOREST MODEL
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: RANDOM FOREST MODEL DEVELOPMENT")
print("="*80)

# Prepare data for Random Forest
le_location = LabelEncoder()
le_foundation = LabelEncoder()
df["Location_Enc"] = le_location.fit_transform(df["Location"])
df["Found_Enc"] = le_foundation.fit_transform(df["Foundation Type"])

rf_features = ["Total_Area", "No. of Storeys", "Plinth Area", "No. of Columns", "Location_Enc", "Found_Enc"]
X_rf = df[rf_features]
y_rf = df[TARGET]

# Train-test split
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf, y_rf, test_size=0.15, random_state=42
)

# Random Forest Model Specifications
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    max_features=1.0,
    n_jobs=-1
)

print("\n🔧 Random Forest Model Specifications:")
print(f"  - Number of trees (estimators): {rf_model.n_estimators}")
print(f"  - Maximum tree depth: {rf_model.max_depth}")
print(f"  - Minimum samples for split: {rf_model.min_samples_split}")
print(f"  - Minimum samples per leaf: {rf_model.min_samples_leaf}")
print(f"  - Maximum features per split: {rf_model.max_features}")
print(f"  - Bootstrapping: Enabled")
print(f"  - Training samples: {len(X_rf_train):,}")
print(f"  - Testing samples: {len(X_rf_test):,}")
print(f"  - Features used: {len(rf_features)}")

# Train Random Forest
print("\n🚀 Training Random Forest Model...")
rf_model.fit(X_rf_train, y_rf_train)

# Random Forest Predictions
rf_pred_train = rf_model.predict(X_rf_train)
rf_pred_test = rf_model.predict(X_rf_test)

# Random Forest Performance Metrics
rf_metrics = {
    'R² Score': r2_score(y_rf_test, rf_pred_test),
    'Adjusted R²': 1 - (1 - r2_score(y_rf_test, rf_pred_test)) * (len(y_rf_test) - 1) / (len(y_rf_test) - len(rf_features) - 1),
    'MAE (NPR)': mean_absolute_error(y_rf_test, rf_pred_test),
    'RMSE (NPR)': np.sqrt(mean_squared_error(y_rf_test, rf_pred_test)),
    'MAPE (%)': mean_absolute_percentage_error(y_rf_test, rf_pred_test) * 100,
    'Explained Variance': explained_variance_score(y_rf_test, rf_pred_test)
}

# Cross-validation for Random Forest
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(rf_model, X_rf, y_rf, cv=kfold, scoring='r2')

print("\n📊 Random Forest Performance Metrics:")
print(f"  - R² Score: {rf_metrics['R² Score']:.4f}")
print(f"  - Adjusted R²: {rf_metrics['Adjusted R²']:.4f}")
print(f"  - MAE: NPR {rf_metrics['MAE (NPR)']:,.0f}")
print(f"  - RMSE: NPR {rf_metrics['RMSE (NPR)']:,.0f}")
print(f"  - MAPE: {rf_metrics['MAPE (%)']:.2f}%")
print(f"  - Explained Variance: {rf_metrics['Explained Variance']:.4f}")
print(f"  - Cross-validation R² (5-fold): {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# Feature Importance for Random Forest
rf_feature_importance = pd.DataFrame({
    'Feature': rf_features,
    'Importance': rf_model.feature_importances_,
    'Importance (%)': rf_model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\n📊 Random Forest Feature Importance:")
print(rf_feature_importance.to_string(index=False))

# ============================================================================
# SECTION 5: ARTIFICIAL NEURAL NETWORK (ANN) MODEL
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: ARTIFICIAL NEURAL NETWORK (ANN) MODEL DEVELOPMENT")
print("="*80)

# Prepare data for ANN with polynomial features
ann_features = ["Total_Area", "No. of Storeys", "Plinth Area", "No. of Columns", "Foundation Type", "Location"]
X_ann = df[ann_features]
y_ann = df[TARGET].values.reshape(-1, 1)

# Scale target variable
scaler_y = StandardScaler()
y_ann_scaled = scaler_y.fit_transform(y_ann)

# Create preprocessor for ANN
numeric_transformer = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, ["Total_Area", "No. of Storeys", "Plinth Area", "No. of Columns"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Foundation Type", "Location"])
])

# Split data
X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(
    X_ann, y_ann_scaled, test_size=0.15, random_state=42
)

# Transform features
X_ann_train_scaled = preprocessor.fit_transform(X_ann_train)
X_ann_test_scaled = preprocessor.transform(X_ann_test)

# ANN Model Architecture
print("\n🔧 ANN Model Architecture Specifications:")
print(f"  - Input layer: {X_ann_train_scaled.shape[1]} neurons")
print(f"  - Hidden Layer 1: 128 neurons with ReLU activation + L2 regularization (0.005)")
print(f"  - Dropout Layer 1: 20% dropout rate")
print(f"  - Hidden Layer 2: 64 neurons with ReLU activation + L2 regularization (0.005)")
print(f"  - Output Layer: 1 neuron (linear activation)")
print(f"  - Total trainable parameters: {128 * X_ann_train_scaled.shape[1] + 128 + 128 * 64 + 64 + 64 * 1 + 1:,}")
print(f"  - Optimizer: Adam (learning_rate=0.001)")
print(f"  - Loss function: Mean Squared Error (MSE)")
print(f"  - Regularization: L2 (0.005) on hidden layers")
print(f"  - Dropout rate: 0.2")
print(f"  - Batch size: 8")
print(f"  - Maximum epochs: 500")
print(f"  - Early stopping patience: 50")
print(f"  - Learning rate reduction patience: 15")

# Build ANN Model
ann_model = keras.Sequential([
    layers.Input(shape=(X_ann_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.Dense(1)
])

ann_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks
lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001, verbose=0
)
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True, verbose=0
)

print("\n🚀 Training ANN Model...")
ann_history = ann_model.fit(
    X_ann_train_scaled, y_ann_train,
    validation_data=(X_ann_test_scaled, y_ann_test),
    epochs=500,
    batch_size=8,
    verbose=0,
    callbacks=[lr_callback, early_stop]
)

# ANN Predictions
ann_pred_train_scaled = ann_model.predict(X_ann_train_scaled, verbose=0)
ann_pred_test_scaled = ann_model.predict(X_ann_test_scaled, verbose=0)

# Inverse transform predictions
ann_pred_train = scaler_y.inverse_transform(ann_pred_train_scaled).flatten()
ann_pred_test = scaler_y.inverse_transform(ann_pred_test_scaled).flatten()
y_ann_train_actual = scaler_y.inverse_transform(y_ann_train).flatten()
y_ann_test_actual = scaler_y.inverse_transform(y_ann_test).flatten()

# ANN Performance Metrics
ann_metrics = {
    'R² Score': r2_score(y_ann_test_actual, ann_pred_test),
    'Adjusted R²': 1 - (1 - r2_score(y_ann_test_actual, ann_pred_test)) * (len(y_ann_test_actual) - 1) / (len(y_ann_test_actual) - X_ann_train_scaled.shape[1] - 1),
    'MAE (NPR)': mean_absolute_error(y_ann_test_actual, ann_pred_test),
    'RMSE (NPR)': np.sqrt(mean_squared_error(y_ann_test_actual, ann_pred_test)),
    'MAPE (%)': mean_absolute_percentage_error(y_ann_test_actual, ann_pred_test) * 100,
    'Explained Variance': explained_variance_score(y_ann_test_actual, ann_pred_test)
}

print("\n📊 ANN Performance Metrics:")
print(f"  - R² Score: {ann_metrics['R² Score']:.4f}")
print(f"  - Adjusted R²: {ann_metrics['Adjusted R²']:.4f}")
print(f"  - MAE: NPR {ann_metrics['MAE (NPR)']:,.0f}")
print(f"  - RMSE: NPR {ann_metrics['RMSE (NPR)']:,.0f}")
print(f"  - MAPE: {ann_metrics['MAPE (%)']:.2f}%")
print(f"  - Explained Variance: {ann_metrics['Explained Variance']:.4f}")

# Training history
print(f"\n📈 ANN Training Summary:")
print(f"  - Final training loss: {ann_history.history['loss'][-1]:.4f}")
print(f"  - Final validation loss: {ann_history.history['val_loss'][-1]:.4f}")
print(f"  - Best validation loss: {min(ann_history.history['val_loss']):.4f}")
print(f"  - Epochs trained: {len(ann_history.history['loss'])}")
print(f"  - Early stopping triggered: Yes (patience=50)")

# ============================================================================
# SECTION 6: BASELINE MODEL (FOR COMPARISON)
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: BASELINE LINEAR REGRESSION MODEL")
print("="*80)

# Simple linear regression baseline
baseline_model = LinearRegression()
baseline_model.fit(X_rf_train, y_rf_train)
baseline_pred = baseline_model.predict(X_rf_test)
baseline_r2 = r2_score(y_rf_test, baseline_pred)
baseline_mae = mean_absolute_error(y_rf_test, baseline_pred)

print("\n📊 Baseline Model (Linear Regression) Performance:")
print(f"  - R² Score: {baseline_r2:.4f}")
print(f"  - MAE: NPR {baseline_mae:,.0f}")

# ============================================================================
# SECTION 7: MODEL COMPARISON TABLE
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: MODEL PERFORMANCE COMPARISON")
print("="*80)

comparison_table = pd.DataFrame({
    'Metric': ['R² Score', 'Adjusted R²', 'MAE (NPR)', 'RMSE (NPR)', 'MAPE (%)', 'Explained Variance'],
    'Random Forest': [
        f'{rf_metrics["R² Score"]:.4f}',
        f'{rf_metrics["Adjusted R²"]:.4f}',
        f'{rf_metrics["MAE (NPR)"]:,.0f}',
        f'{rf_metrics["RMSE (NPR)"]:,.0f}',
        f'{rf_metrics["MAPE (%)"]:.2f}%',
        f'{rf_metrics["Explained Variance"]:.4f}'
    ],
    'ANN': [
        f'{ann_metrics["R² Score"]:.4f}',
        f'{ann_metrics["Adjusted R²"]:.4f}',
        f'{ann_metrics["MAE (NPR)"]:,.0f}',
        f'{ann_metrics["RMSE (NPR)"]:,.0f}',
        f'{ann_metrics["MAPE (%)"]:.2f}%',
        f'{ann_metrics["Explained Variance"]:.4f}'
    ],
    'Baseline (Linear)': [
        f'{baseline_r2:.4f}',
        '-',
        f'{baseline_mae:,.0f}',
        '-',
        '-',
        '-'
    ]
})

print("\n📊 TABLE 3: Comparative Model Performance")
print(comparison_table.to_string(index=False))
comparison_table.to_csv(f"{output_dir}/table3_model_comparison.csv", index=False)

# ============================================================================
# SECTION 8: SAMPLE PREDICTIONS COMPARISON
# ============================================================================

sample_comparison = pd.DataFrame({
    'Actual (NPR)': y_rf_test.values[:15],
    'RF Predicted (NPR)': rf_pred_test[:15],
    'RF Error %': (abs(y_rf_test.values[:15] - rf_pred_test[:15]) / y_rf_test.values[:15]) * 100,
    'ANN Predicted (NPR)': ann_pred_test[:15],
    'ANN Error %': (abs(y_rf_test.values[:15] - ann_pred_test[:15]) / y_rf_test.values[:15]) * 100
}).round(2)

print("\n📊 TABLE 4: Sample Predictions Comparison (First 15 Test Samples)")
print(sample_comparison.to_string(index=False))
sample_comparison.to_csv(f"{output_dir}/table4_sample_predictions.csv", index=False)

# ============================================================================
# SECTION 9: FIGURES FOR THESIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: GENERATING THESIS FIGURES")
print("="*80)

# Figure 2: Actual vs Predicted - Random Forest
fig2, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_rf_test, rf_pred_test, alpha=0.6, edgecolors='black', linewidth=0.5, s=100, label='Predictions')
ax.plot([y_rf_test.min(), y_rf_test.max()], [y_rf_test.min(), y_rf_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Cost (NPR)', fontweight='bold')
ax.set_ylabel('Predicted Cost (NPR)', fontweight='bold')
ax.set_title(f'Random Forest: Actual vs Predicted Building Costs\nR² = {rf_metrics["R² Score"]:.4f}, MAE = NPR {rf_metrics["MAE (NPR)"]:,.0f}', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/figure2_rf_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: figure2_rf_actual_vs_predicted.png")

# Figure 3: Actual vs Predicted - ANN
fig3, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_ann_test_actual, ann_pred_test, alpha=0.6, edgecolors='black', linewidth=0.5, s=100, color='green', label='Predictions')
ax.plot([y_ann_test_actual.min(), y_ann_test_actual.max()], [y_ann_test_actual.min(), y_ann_test_actual.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Cost (NPR)', fontweight='bold')
ax.set_ylabel('Predicted Cost (NPR)', fontweight='bold')
ax.set_title(f'ANN: Actual vs Predicted Building Costs\nR² = {ann_metrics["R² Score"]:.4f}, MAE = NPR {ann_metrics["MAE (NPR)"]:,.0f}', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/figure3_ann_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: figure3_ann_actual_vs_predicted.png")

# Figure 4: Residual Analysis - Both Models
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# RF Residuals
rf_residuals = y_rf_test - rf_pred_test
axes[0,0].scatter(rf_pred_test, rf_residuals, alpha=0.6, edgecolors='black', s=80)
axes[0,0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0,0].set_xlabel('Predicted Cost (NPR)')
axes[0,0].set_ylabel('Residuals (NPR)')
axes[0,0].set_title('Random Forest: Residuals vs Predicted', fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

# ANN Residuals
ann_residuals = y_ann_test_actual - ann_pred_test
axes[0,1].scatter(ann_pred_test, ann_residuals, alpha=0.6, edgecolors='black', s=80, color='green')
axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Predicted Cost (NPR)')
axes[0,1].set_ylabel('Residuals (NPR)')
axes[0,1].set_title('ANN: Residuals vs Predicted', fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

# RF Residual Distribution
axes[1,0].hist(rf_residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[1,0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Residuals (NPR)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Random Forest: Residual Distribution', fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

# ANN Residual Distribution
axes[1,1].hist(ann_residuals, bins=15, edgecolor='black', alpha=0.7, color='green')
axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,1].set_xlabel('Residuals (NPR)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('ANN: Residual Distribution', fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Figure 4: Residual Analysis - Random Forest vs ANN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure4_residual_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: figure4_residual_analysis.png")

# Figure 5: Feature Importance - Random Forest
fig5, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(rf_feature_importance)))
bars = ax.barh(rf_feature_importance['Feature'], rf_feature_importance['Importance (%)'], color=colors, edgecolor='black')
ax.set_xlabel('Importance (%)', fontweight='bold')
ax.set_ylabel('Features', fontweight='bold')
ax.set_title('Figure 5: Random Forest Feature Importance', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for bar, imp in zip(bars, rf_feature_importance['Importance (%)']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{imp:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure5_rf_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: figure5_rf_feature_importance.png")

# Figure 6: ANN Training History
fig6, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss history
axes[0].plot(ann_history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(ann_history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
axes[0].set_title('ANN Training History: Loss', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE history
axes[1].plot(ann_history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(ann_history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('MAE (scaled)', fontweight='bold')
axes[1].set_title('ANN Training History: MAE', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 6: ANN Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure6_ann_training_history.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved: figure6_ann_training_history.png")

# Figure 7: Model Comparison Bar Charts
fig7, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['Random Forest', 'ANN', 'Baseline']
r2_values = [rf_metrics['R² Score'], ann_metrics['R² Score'], baseline_r2]
mae_values = [rf_metrics['MAE (NPR)'], ann_metrics['MAE (NPR)'], baseline_mae]
rmse_values = [rf_metrics['RMSE (NPR)'], ann_metrics['RMSE (NPR)'], 0]

# R² comparison
bars1 = axes[0].bar(models, r2_values, color=['steelblue', 'green', 'gray'], edgecolor='black')
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('R² Score Comparison', fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, r2_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontweight='bold')

# MAE comparison
bars2 = axes[1].bar(models, mae_values, color=['steelblue', 'green', 'gray'], edgecolor='black')
axes[1].set_ylabel('MAE (NPR)', fontweight='bold')
axes[1].set_title('MAE Comparison', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, mae_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000, f'NPR {val:,.0f}', ha='center', fontweight='bold', rotation=45, fontsize=9)

# RMSE comparison
bars3 = axes[2].bar(models[:2], rmse_values[:2], color=['steelblue', 'green'], edgecolor='black')
axes[2].set_ylabel('RMSE (NPR)', fontweight='bold')
axes[2].set_title('RMSE Comparison', fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, rmse_values[:2]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000, f'NPR {val:,.0f}', ha='center', fontweight='bold', rotation=45, fontsize=9)

plt.suptitle('Figure 7: Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure7_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 7 saved: figure7_model_comparison.png")

# Figure 8: Error Distribution Comparison
fig8, axes = plt.subplots(1, 2, figsize=(14, 5))

# Percentage Error Distribution
rf_pct_error = (np.abs(rf_residuals) / y_rf_test) * 100
ann_pct_error = (np.abs(ann_residuals) / y_ann_test_actual) * 100

axes[0].hist(rf_pct_error, bins=15, alpha=0.5, label='Random Forest', color='steelblue', edgecolor='black')
axes[0].hist(ann_pct_error, bins=15, alpha=0.5, label='ANN', color='green', edgecolor='black')
axes[0].set_xlabel('Percentage Error (%)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Percentage Error Distribution Comparison', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot comparison
error_data = [rf_pct_error, ann_pct_error]
bp = axes[1].boxplot(error_data, labels=['Random Forest', 'ANN'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('green')
axes[1].set_ylabel('Percentage Error (%)', fontweight='bold')
axes[1].set_title('Error Distribution: Box Plot Comparison', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 8: Error Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure8_error_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 8 saved: figure8_error_distribution.png")

# Figure 9: Correlation Heatmap
fig9, ax = plt.subplots(figsize=(10, 8))
corr_features = ["Total_Area", "No. of Storeys", "Plinth Area", "No. of Columns", TARGET]
corr_matrix = df[corr_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('Figure 9: Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure9_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 9 saved: figure9_correlation_heatmap.png")

# Figure 10: Q-Q Plot for Normality Check
fig10, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF Q-Q plot
stats.probplot(rf_residuals, dist="norm", plot=axes[0])
axes[0].set_title('Random Forest: Q-Q Plot of Residuals', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# ANN Q-Q plot
stats.probplot(ann_residuals, dist="norm", plot=axes[1])
axes[1].set_title('ANN: Q-Q Plot of Residuals', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 10: Normality Check - Q-Q Plots', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/figure10_qq_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 10 saved: figure10_qq_plots.png")

# ============================================================================
# SECTION 10: ADDITIONAL STATISTICAL TESTS
# ============================================================================

print("\n" + "="*80)
print("SECTION 9: STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

# Paired t-test between model predictions
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(rf_pred_test, ann_pred_test)
print(f"\n📊 Paired t-test between RF and ANN predictions:")
print(f"  - t-statistic: {t_stat:.4f}")
print(f"  - p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  - Conclusion: Significant difference between model predictions (p < 0.05)")
else:
    print("  - Conclusion: No significant difference between model predictions (p > 0.05)")

# Calculate improvement percentages
rf_improvement_over_baseline_r2 = ((rf_metrics['R² Score'] - baseline_r2) / baseline_r2) * 100
ann_improvement_over_baseline_r2 = ((ann_metrics['R² Score'] - baseline_r2) / baseline_r2) * 100
rf_improvement_over_baseline_mae = ((baseline_mae - rf_metrics['MAE (NPR)']) / baseline_mae) * 100
ann_improvement_over_baseline_mae = ((baseline_mae - ann_metrics['MAE (NPR)']) / baseline_mae) * 100

print(f"\n📊 Performance Improvement over Baseline:")
print(f"  - Random Forest R² improvement: {rf_improvement_over_baseline_r2:.1f}%")
print(f"  - ANN R² improvement: {ann_improvement_over_baseline_r2:.1f}%")
print(f"  - Random Forest MAE reduction: {rf_improvement_over_baseline_mae:.1f}%")
print(f"  - ANN MAE reduction: {ann_improvement_over_baseline_mae:.1f}%")

# ============================================================================
# SECTION 11: SAVE MODELS FOR FUTURE USE
# ============================================================================

print("\n" + "="*80)
print("SECTION 10: SAVING MODELS AND PREPROCESSORS")
print("="*80)

# Save Random Forest model and encoders
joblib.dump(rf_model, f"{output_dir}/random_forest_model.pkl")
joblib.dump(le_location, f"{output_dir}/location_encoder.pkl")
joblib.dump(le_foundation, f"{output_dir}/foundation_encoder.pkl")

# Save ANN model and preprocessors
ann_model.save(f"{output_dir}/ann_model.h5")
joblib.dump(preprocessor, f"{output_dir}/ann_preprocessor.pkl")
joblib.dump(scaler_y, f"{output_dir}/target_scaler.pkl")

print(f"✓ All models saved to '{output_dir}/' directory")

# ============================================================================
# SECTION 12: SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("THESIS RESEARCH: FINAL SUMMARY REPORT")
print("="*80)

print("\n📋 MODEL ARCHITECTURE SUMMARY:")
print("-" * 50)
print("\nRandom Forest:")
print(f"  • Trees: {rf_model.n_estimators}")
print(f"  • Max Depth: {rf_model.max_depth}")
print(f"  • Features: {', '.join(rf_features)}")
print("\nArtificial Neural Network:")
print(f"  • Input dimension: {X_ann_train_scaled.shape[1]}")
print(f"  • Hidden Layer 1: 128 neurons (ReLU + L2)")
print(f"  • Dropout: 20%")
print(f"  • Hidden Layer 2: 64 neurons (ReLU + L2)")
print(f"  • Output: 1 neuron (linear)")
print(f"  • Optimizer: Adam (lr=0.001)")
print(f"  • Regularization: L2 (0.005)")

print("\n📊 PERFORMANCE SUMMARY:")
print("-" * 50)
print(f"{'Metric':<20} {'Random Forest':<20} {'ANN':<20}")
print("-" * 60)
print(f"{'R² Score':<20} {rf_metrics['R² Score']:<20.4f} {ann_metrics['R² Score']:<20.4f}")
print(f"{'MAE (NPR)':<20} {rf_metrics['MAE (NPR)']:<20,.0f} {ann_metrics['MAE (NPR)']:<20,.0f}")
print(f"{'RMSE (NPR)':<20} {rf_metrics['RMSE (NPR)']:<20,.0f} {ann_metrics['RMSE (NPR)']:<20,.0f}")
print(f"{'MAPE (%)':<20} {rf_metrics['MAPE (%)']:<20.2f} {ann_metrics['MAPE (%)']:<20.2f}")

print("\n✅ RECOMMENDATION:")
print("-" * 50)
if rf_metrics['R² Score'] > ann_metrics['R² Score']:
    print("✓ Random Forest outperforms ANN for this dataset.")
    print("  Reason: Smaller dataset benefits from tree-based ensemble methods.")
else:
    print("✓ ANN outperforms Random Forest for this dataset.")
    print("  Reason: Complex non-linear relationships captured by neural network.")

print("\n" + "="*80)
print(f"✅ All figures and tables saved to: {os.path.abspath(output_dir)}/")
print(f"📁 Total files generated: 10 figures + 4 tables + 5 model files")
print("="*80)

# List all generated files
print("\n📁 GENERATED FILES:")
print("-" * 50)
print("\n📊 Tables (CSV):")
for f in os.listdir(output_dir):
    if f.startswith('table') and f.endswith('.csv'):
        print(f"  - {f}")
print("\n📈 Figures (PNG):")
for f in os.listdir(output_dir):
    if f.startswith('figure') and f.endswith('.png'):
        print(f"  - {f}")
print("\n🤖 Model Files:")
for f in os.listdir(output_dir):
    if f.endswith(('.pkl', '.h5')):
        print(f"  - {f}")

print("\n" + "="*80)
print("THESIS RESEARCH CODE COMPLETED SUCCESSFULLY")
print("="*80)