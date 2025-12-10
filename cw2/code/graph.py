
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ═══════════════════════════════════════════════════════════════
# METRICS VISUALIZATION - Multiple Views
# ═══════════════════════════════════════════════════════════════

df_metrics = pd.read_csv("/Users/sarthakpunj/Desktop/cw2/evaluation_results/metrics.csv")

# ────────────────────────────────────────────────────────────────
# Graph 1a: Grouped Bar Chart (Your Original - Improved)
# ────────────────────────────────────────────────────────────────

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
models = df_metrics["Model"].tolist()
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

bar_width = 0.15
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(14, 7))

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i * bar_width,
                   df_metrics[metric],
                   width=bar_width,
                   label=metric,
                   color=color,
                   edgecolor='black',
                   linewidth=1.2)
    
    # Add value labels on bars
    for j, (bar, value) in enumerate(zip(bars, df_metrics[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold')

ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison Across All Metrics\n(Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + bar_width * 2)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.set_ylim(0.9, 1.02)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_1_metrics_bar.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 1a saved: graph_1_metrics_bar.png")

# ────────────────────────────────────────────────────────────────
# Graph 1b: Heatmap - Color-Coded Performance Matrix
# ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for heatmap
heatmap_data = df_metrics[metrics].T
heatmap_data.columns = models

# Create heatmap
sns.heatmap(heatmap_data, 
            annot=True, 
            fmt='.4f',
            cmap='RdYlGn',
            vmin=0.90, 
            vmax=1.0,
            cbar_kws={'label': 'Score'},
            linewidths=2,
            linecolor='white',
            ax=ax)

ax.set_title('Model Performance Heatmap\n(Higher = Better)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Metrics', fontsize=13, fontweight='bold')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_1_metrics_heatmap.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 1b saved: graph_1_metrics_heatmap.png")

# ────────────────────────────────────────────────────────────────
# Graph 1c: Radar/Spider Chart - All Metrics at Once
# ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors_radar = ['#3498db', '#e74c3c', '#2ecc71']

for i, model in enumerate(models):
    values = df_metrics.loc[df_metrics['Model'] == model, metrics].values.flatten().tolist()
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[i])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[i])
    
    # Add value labels
    for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
        ax.text(angle, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.set_ylim(0.90, 1.02)
ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.00])
ax.set_yticklabels(['0.92', '0.94', '0.96', '0.98', '1.00'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

ax.set_title('Model Performance - Radar Chart\nAll Metrics Comparison', 
             fontsize=16, fontweight='bold', pad=30, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_1_metrics_radar.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 1c saved: graph_1_metrics_radar.png")

# ═══════════════════════════════════════════════════════════════
#  OVERFITTING ANALYSIS - Multiple Views
# ═══════════════════════════════════════════════════════════════

df_overfit = pd.read_csv("/Users/sarthakpunj/Desktop/cw2/evaluation_results/overfitting_check.csv")

models_list = df_overfit["Model"].unique()
datasets = ["Train", "Val", "Test"]

# ────────────────────────────────────────────────────────────────
# Graph 2a: Grouped Bar Chart (Improved)
# ────────────────────────────────────────────────────────────────

bar_width = 0.25
x = np.arange(len(models_list))
colors_datasets = ['#3498db', '#e74c3c', '#2ecc71']

fig, ax = plt.subplots(figsize=(12, 7))

for i, (dataset, color) in enumerate(zip(datasets, colors_datasets)):
    subset = df_overfit[df_overfit["Dataset"] == dataset]
    bars = ax.bar(x + i * bar_width,
                   subset["Accuracy"],
                   width=bar_width,
                   label=dataset,
                   color=color,
                   edgecolor='black',
                   linewidth=1.2)
    
    # Add value labels
    for j, (bar, value) in enumerate(zip(bars, subset["Accuracy"])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Overfitting Check: Accuracy Across Train/Val/Test\n(Gap < 0.01% = No Overfitting)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(models_list, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.9)
ax.set_ylim(0.955, 0.960)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation
ax.text(0.5, 0.02, ' All gaps < 0.01% → Excellent Generalization!', 
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_2_overfitting_bar.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 2a saved: graph_2_overfitting_bar.png")

# ────────────────────────────────────────────────────────────────
# Graph 2b: Line Chart - Consistency Visualization
# ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

for i, model in enumerate(models_list):
    model_data = df_overfit[df_overfit["Model"] == model]
    accuracies = model_data["Accuracy"].values
    
    ax.plot(datasets, accuracies, 
            marker='o', markersize=12, linewidth=3,
            label=model, color=colors_datasets[i])
    
    # Add value labels
    for j, (dataset, acc) in enumerate(zip(datasets, accuracies)):
        ax.text(j, acc + 0.00005, f'{acc:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Generalization: Train → Val → Test\n(Flat Lines = Perfect Generalization)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, framealpha=0.9, loc='lower right')
ax.set_ylim(0.9575, 0.9582)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_2_overfitting_lines.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 2b saved: graph_2_overfitting_lines.png")

# ────────────────────────────────────────────────────────────────
# Graph 2c: Gap Analysis - Visual Difference
# ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

gaps = []
gap_labels = []

for model in models_list:
    model_data = df_overfit[df_overfit["Model"] == model]
    train_acc = model_data[model_data["Dataset"] == "Train"]["Accuracy"].values[0]
    test_acc = model_data[model_data["Dataset"] == "Test"]["Accuracy"].values[0]
    gap = abs(train_acc - test_acc) * 100  # Convert to percentage
    gaps.append(gap)
    gap_labels.append(f'{model}\n({gap:.3f}%)')

colors_gap = ['lightgreen' if g < 0.02 else 'yellow' for g in gaps]

bars = ax.barh(gap_labels, gaps, color=colors_gap, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, gap in zip(bars, gaps):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{gap:.3f}%',
            ha='left', va='center',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Gap (Train - Test Accuracy) %', fontsize=13, fontweight='bold')
ax.set_title('Overfitting Gap Analysis\n(Lower = Better, < 2% = Excellent)', 
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0.02, color='red', linestyle='--', linewidth=2, label='Threshold (2%)')
ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Warning (5%)')
ax.legend(fontsize=11)
ax.set_xlim(0, 0.1)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_2_overfitting_gaps.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 2c saved: graph_2_overfitting_gaps.png")

# ═══════════════════════════════════════════════════════════════
# COMBINED ANALYSIS - Side-by-Side Comparison
# ═══════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: F1-Score comparison
metrics_to_show = ["Accuracy", "F1-Score"]
x_pos = np.arange(len(models))

for i, metric in enumerate(metrics_to_show):
    bars = ax1.bar(x_pos + i * 0.35,
                    df_metrics[metric],
                    width=0.35,
                    label=metric,
                    edgecolor='black',
                    linewidth=1.2)
    
    for bar, value in zip(bars, df_metrics[metric]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{value:.4f}',
                 ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

ax1.set_xticks(x_pos + 0.175)
ax1.set_xticklabels(models, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Test Performance\n(Accuracy & F1-Score)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_ylim(0.95, 1.01)
ax1.grid(axis='y', alpha=0.3)

# Right: Overfitting comparison
x_pos2 = np.arange(len(models_list))

for i, (dataset, color) in enumerate(zip(datasets, colors_datasets)):
    subset = df_overfit[df_overfit["Dataset"] == dataset]
    bars = ax2.bar(x_pos2 + i * 0.25,
                    subset["Accuracy"],
                    width=0.25,
                    label=dataset,
                    color=color,
                    edgecolor='black',
                    linewidth=1.2)
    
    for bar, value in zip(bars, subset["Accuracy"]):
        height = bar.get_height()
        if i == 1:  # Only show labels on middle bars to avoid clutter
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                     f'{value:.4f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

ax2.set_xticks(x_pos2 + 0.25)
ax2.set_xticklabels(models_list, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Generalization Check\n(Train/Val/Test Consistency)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_ylim(0.955, 0.960)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Complete Model Evaluation Summary', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/sarthakpunj/Desktop/cw2/evaluation_results/graph_3_combined_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(" Graph 3 saved: graph_3_combined_analysis.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(" ALL GRAPHS GENERATED SUCCESSFULLY!")
print("="*60)
print("\n Saved in: /Users/sarthakpunj/Desktop/cw2/evaluation_results/\n")
print("Metrics Visualizations:")
print("  1. graph_1_metrics_bar.png       - Grouped bar chart")
print("  2. graph_1_metrics_heatmap.png   - Color-coded heatmap")
print("  3. graph_1_metrics_radar.png     - Spider/radar chart")
print("\nOverfitting Analysis:")
print("  4. graph_2_overfitting_bar.png   - Train/Val/Test bars")
print("  5. graph_2_overfitting_lines.png - Consistency lines")
print("  6. graph_2_overfitting_gaps.png  - Gap analysis")
print("\nCombined:")
print("  7. graph_3_combined_analysis.png - Side-by-side summary")
print("\n" + "="*60)
print(" Ready for presentation/report!")
print("="*60)