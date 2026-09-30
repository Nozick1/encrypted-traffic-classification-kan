"""Merge the completed MLP run + resumed KAN run into one complete summary.

The full training was split across two processes (MLP 10/10 done, then KAN
resumed 6->10), so seed0/summary.json only holds `kan`. This script re-evaluates
both best weights on the test set (confusion matrix + test metrics), pulls MLP's
history from checkpoint_mlp.json, and writes a unified summary + final_metrics.
"""
import json
import os

import torch

from train_compare import (build_model, build_loaders, evaluate, _metrics,
                           CLASS_NAMES)

HERE = os.path.dirname(__file__)
SEED0 = os.path.join(HERE, 'results', 'seed0')
RESULTS = os.path.join(HERE, 'results')
DATA_DIR = r'C:\Users\秦宇欣\Desktop\2022-2026\stitp\kan\kan\k375训练(1)\k375训练\dataset'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    _, test_loader = build_loaders(DATA_DIR, batch_size=32, num_workers=0)

    # KAN part already in seed0/summary.json (written by the resumed run).
    with open(os.path.join(SEED0, 'summary.json'), encoding='utf-8') as f:
        kan = json.load(f)['kan']

    # MLP full history + best val acc from its checkpoint.
    with open(os.path.join(SEED0, 'checkpoint_mlp.json'), encoding='utf-8') as f:
        mlp_ckpt = json.load(f)

    summary = {}
    final_metrics = {}
    for name in ['mlp', 'kan']:
        model = build_model(name).to(device)
        model.load_state_dict(
            torch.load(os.path.join(SEED0, f'best_{name}.pth'), weights_only=True))
        n_params = sum(p.numel() for p in model.parameters())

        loss, acc, f1, prec, rec, y_true, y_pred = evaluate(model, test_loader, device)
        _, _, _, cm = _metrics(y_true, y_pred)

        if name == 'mlp':
            history, best_val_acc = mlp_ckpt['history'], mlp_ckpt['best_acc']
        else:
            history, best_val_acc = kan['history'], kan['best_val_acc']

        summary[name] = {
            'best_val_acc': best_val_acc,
            'n_params': n_params,
            'history': history,
            'confusion_matrix': cm.tolist(),
            'class_names': CLASS_NAMES,
        }
        final_metrics[name] = {
            'test_loss': loss,
            'test_acc': acc,
            'test_f1': f1,
            'test_precision': prec,
            'test_recall': rec,
            'confusion_matrix': cm.tolist(),
            'class_names': CLASS_NAMES,
        }
        print(f'[{name}] test_acc={acc:.4f} f1={f1:.4f} '
              f'precision={prec:.4f} recall={rec:.4f}')

    summary['_meta'] = {'seed': 0, 'epochs': 10, 'batch': 64, 'lr': 0.001,
                        'subset': None}

    for path in (os.path.join(SEED0, 'summary.json'),
                 os.path.join(RESULTS, 'summary.json')):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print('wrote', path)

    with open(os.path.join(RESULTS, 'final_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print('wrote', os.path.join(RESULTS, 'final_metrics.json'))


if __name__ == '__main__':
    main()
