"""Lightweight evaluation of the saved best models (no training, no gradients).

Recovers final test metrics after the training process was stopped.
"""
import json
import os

import numpy as np
import torch

from train_compare import (build_model, build_loaders, evaluate, _metrics,
                           CLASS_NAMES)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, 'results')


def main():
    data_dir = r'C:\Users\秦宇欣\Desktop\2022-2026\stitp\kan\kan\k375训练(1)\k375训练\dataset'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    # num_workers=0 keeps memory low (single process, forward-only).
    _, test_loader = build_loaders(data_dir, batch_size=32, num_workers=0)

    out = {}
    for name in ['mlp', 'kan']:
        ckpt = os.path.join(RESULTS, f'best_{name}.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] {ckpt} not found')
            continue
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        loss, acc, f1, prec, rec, y_true, y_pred = evaluate(model, test_loader, device)
        _, _, _, cm = _metrics(y_true, y_pred)
        out[name] = {
            'test_loss': loss, 'test_acc': acc, 'test_f1': f1,
            'test_precision': prec, 'test_recall': rec,
            'confusion_matrix': cm.tolist(), 'class_names': CLASS_NAMES,
        }
        print(f'\n=== {name.upper()} ===')
        print(f'test_acc={acc:.4f}  f1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}')

        # per-class accuracy
        per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1e-12)
        for i, c in enumerate(CLASS_NAMES):
            print(f'  {c:16s} acc={per_class[i]:.4f}')

    with open(os.path.join(RESULTS, 'final_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\nSaved ->', os.path.join(RESULTS, 'final_metrics.json'))


if __name__ == '__main__':
    main()
