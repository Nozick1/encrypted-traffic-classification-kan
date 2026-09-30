"""Generate figures from results/final_metrics.json (the recovered test results)."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, 'results')
MODEL_LABEL = {'mlp': 'MLP head', 'kan': 'ChebyKAN head'}


def main():
    with open(os.path.join(RESULTS, 'final_metrics.json'), encoding='utf-8') as f:
        data = json.load(f)

    names = [k for k in ['mlp', 'kan'] if k in data]
    cls = data[names[0]]['class_names']

    # ---- Confusion matrices ----
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(data[name]['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(cls, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{MODEL_LABEL[name]}  (test acc {data[name]["test_acc"]:.3f})')
        thresh = cm.max() / 2
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                        color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.suptitle('Confusion matrix — same CNN features, different classifier head')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'confusion_matrices.png'), dpi=150)
    plt.close(fig)

    # ---- Bar chart: acc / f1 / precision ----
    metrics = ['test_acc', 'test_f1', 'test_precision']
    labels = ['Accuracy', 'F1', 'Precision']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics))
    w = 0.35
    for off, name in zip([-w / 2, w / 2], names):
        vals = [data[name][m] for m in metrics]
        ax.bar(x + off, vals, w, label=MODEL_LABEL[name],
               color='#2c7fb8' if name == 'mlp' else '#d95f02')
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title('MLP head vs ChebyKAN head (identical CNN features)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'summary_bar.png'), dpi=150)
    plt.close(fig)

    print('Figures ->', RESULTS)


if __name__ == '__main__':
    main()
