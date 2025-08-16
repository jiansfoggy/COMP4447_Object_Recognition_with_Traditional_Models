import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

# bar chart to show two models' performance on each category
title_info = {"SVC":"C=10","RFC":"n_estimators=300"}
color_info = {"SVC":"skyblue","RFC":"lightcoral"}
precision_dict = {
    'SVC': {
        'airplane': 0.61, 'automobile': 0.63, 'bird': 0.46,
        'cat': 0.37, 'deer': 0.51, 'dog': 0.46,
        'frog': 0.62, 'horse': 0.64, 'ship': 0.70, 'truck': 0.61
    },
    'RFC': {
        'airplane': 0.55, 'automobile': 0.52, 'bird': 0.41,
        'cat': 0.36, 'deer': 0.42, 'dog': 0.43,
        'frog': 0.46, 'horse': 0.51, 'ship': 0.58, 'truck': 0.47
    }
}

models = list(precision_dict.keys())

# Create figure with two horizontal subplots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, model_name in zip((ax0, ax1), models):
    data = precision_dict[model_name]
    classes = list(data.keys())
    precisions = [data[c] for c in classes]
    bars = ax.bar(classes, precisions, color=color_info[model_name])
    ax.set_ylim(0, 1)
    ax.set_title(f'Accuracy by Class: {model_name} | {title_info[model_name]}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, p in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.02,
                f'{p:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

output_path = f'./{models[0]}_{models[1]}_acc.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved entire figure to {output_path}")

# plot confusion matrix for two models in heatmap
cm_svc = np.array([
    [654,  31,  58,  30,  24,  12,  21,  23,  92,  55],
    [ 46, 697,  12,  23,   7,  18,   7,  12,  44, 134],
    [ 71,  29, 480,  83, 102,  70,  78,  57,  13,  17],
    [ 24,  32,  95, 385,  57, 199,  91,  44,  28,  45],
    [ 50,  12, 159,  67, 465,  63,  77,  69,  25,  13],
    [ 26,  17,  75, 224,  60, 438,  54,  62,  22,  22],
    [ 16,  20,  89, 100,  98,  39, 587,  23,  10,  18],
    [ 38,  29,  51,  69,  68,  83,  16, 588,  11,  47],
    [101,  67,  15,  31,  17,  12,   6,  13, 685,  53],
    [ 47, 168,   9,  31,  12,  26,  11,  29,  43, 624]])

cm_rfc = np.array([
    [559,  42,  51,  20,  22,  16,  22,  30, 170,  68],
    [ 24, 566,  17,  33,  16,  24,  43,  33,  67, 177],
    [106,  49, 340,  62, 124,  68, 131,  60,  29,  31],
    [ 53,  46,  68, 262,  72, 194, 141,  58,  24,  82],
    [ 52,  22, 133,  46, 383,  49, 169,  94,  28,  24],
    [ 31,  35,  80, 147,  69, 403,  86,  88,  27,  34],
    [  8,  30,  73,  68, 105,  58, 580,  27,   4,  47],
    [ 48,  43,  40,  47,  92,  78,  47, 479,  22, 104],
    [ 88,  89,  16,  20,  17,  36,  10,  21, 625,  78],
    [ 39, 165,  17,  23,  12,  18,  30,  42,  73, 581]])

cm = {"SVC": cm_svc, "RFC": cm_rfc}
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

sns.set(font_scale=1.0)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

for ax, (model_name, matrix) in zip(axes, cm.items()):
    # Normalize by column (i.e., predicted label totals)
    cm_norm = matrix.astype('float') / matrix.sum(axis=0)[np.newaxis, :]

    sns.heatmap(cm_norm,annot=True,fmt=".2%",cmap="Blues",
        xticklabels=class_names,yticklabels=class_names,
        linewidths=0.5,cbar=True,ax=ax
    )

    ax.set_title(f'{model_name} - Normalized by Column')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

output_path = "svc_rfc_confusion_matrix.png"
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"Saved entire figure to {output_path}")

# bar plot to compare the model accuracy by hyperparameters
precision_dict = {
    'SVC': {
        "0.01": 0.3612, "0.1": 0.4569, "0.5": 0.5179,
        "0.7": 0.5305, "1": 0.5434, "10": 0.5603,
        "15": 0.5564, "20": 0.5548
    },
    'RFC': {
        '10': 0.3517, '50': 0.4375, '100': 0.4595,
        '120': 0.4645, '150': 0.4643,
        '200': 0.4712, '250': 0.472, '300': 0.4778
    }
}

title_info = {"SVC":"C","RFC":"n_estimators"}

models = list(precision_dict.keys())

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, model in zip(axes, models):
    params = list(precision_dict[model].keys())
    scores = list(precision_dict[model].values())
    x = np.arange(len(params))
    bars = ax.bar(x, scores, color=color_info[model])
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylim(0, 1)
    ax.set_title(f'{model}: Tune parameter {title_info[model]}')
    ax.set_xlabel(model)
    if ax is axes[0]:
        ax.set_ylabel('Accuracy')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # annotate bars
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                sc + 0.01,
                f'{sc:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save the figure with both subplots
output_path = 'precision_svc_rfc.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved figure to {output_path}")
