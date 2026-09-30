"""Generate paper-ready figures from results/summary.json."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'summary.json')
OUT = os.path.join(os.path.dirname(__file__), 'results')

MODEL_LABEL = {'mlp': 'MLP head', 'kan': 'ChebyKAN head'}
COLORS = {'mlp': '#2c7fb8', 'kan': '#d95f02'}


def main():
    with open(RESULTS, encoding='utf-8') as f:
        summary = json.load(f)

    names = [k for k in ['mlp', 'kan'] if k in summary]

    # ---- Training curves: accuracy & loss ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        h = summary[name]['history']
        ep = np.arange(1, len(h['train_acc']) + 1)
        c = COLORS[name]
        axes[0].plot(ep, h['train_acc'], '-', color=c, alpha=0.4,
                     label=f'{MODEL_LABEL[name]} train')
        axes[0].plot(ep, h['val_acc'], '-o', color=c,
                     label=f'{MODEL_LABEL[name]} val')
        axes[1].plot(ep, h['train_loss'], '-', color=c, alpha=0.4,
                     label=f'{MODEL_LABEL[name]} train')
        axes[1].plot(ep, h['val_loss'], '-o', color=c,
                     label=f'{MODEL_LABEL[name]} val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle('CNN feature extractor: MLP vs ChebyKAN head')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'training_curves.png'), dpi=150)
    plt.close(fig)

    # ---- Confusion matrices ----
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(summary[name]['confusion_matrix'])
        cls = summary[name]['class_names']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(cls, fontsize=7)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(MODEL_LABEL[name])
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=6, color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.suptitle('Confusion matrix (test set)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'confusion_matrices.png'), dpi=150)
    plt.close(fig)

    # ---- Summary bar chart ----
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    w = 0.35
    best_acc = [summary[n]['best_val_acc'] for n in names]
    final_f1 = [summary[n]['history']['val_f1'][-1] for n in names]
    ax.bar(x - w / 2, best_acc, w, label='Best val acc', color='#2c7fb8')
    ax.bar(x + w / 2, final_f1, w, label='Final val F1', color='#d95f02')
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[n] for n in names])
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    for i, v in enumerate(best_acc):
        ax.text(i - w / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(final_f1):
        ax.text(i + w / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title('MLP head vs ChebyKAN head (same CNN features)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'summary_bar.png'), dpi=150)
    plt.close(fig)

    print('Figures written to', OUT)


if __name__ == '__main__':
    main()
