"""Fair comparison: CNN feature extractor + MLP head  vs  + ChebyKAN head.

Both models share the *same* 3-conv feature extractor and the *same* training
recipe; only the classification head differs. This isolates the MLP-vs-KAN
question cleanly.

Usage:
  python train_compare.py --model mlp            # baseline
  python train_compare.py --model kan            # KAN head
  python train_compare.py --model both           # run both back-to-back
  python train_compare.py --model both --subset 3000 --epochs 1   # quick smoke
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from chebykan import ChebyKAN_Head


def _metrics(y_true, y_pred, n_classes=10):
    """Weighted macro metrics + confusion matrix, computed with numpy only."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    w = support / support.sum()
    return (float((f1 * w).sum()), float((prec * w).sum()),
            float((rec * w).sum()), cm)

CLASS_NAMES = ['BitTorrent', 'Facetime', 'Gmail', 'Miuref', 'MySQL',
               'Outlook', 'Skype', 'Tinba', 'WorldOfWarcraft', 'Zeus']


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(data_dir, batch_size, num_workers, subset=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
    if subset is not None and subset < len(train_ds):
        # deterministic subset for smoke tests
        idx = torch.randperm(len(train_ds))[:subset].tolist()
        train_ds = Subset(train_ds, idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.gap(x)          # -> (B, 128, 1, 1)
        return x.flatten(1)      # -> (B, 128)


class MLPHead(nn.Module):
    def __init__(self, in_dim=128, hidden=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.head = MLPHead()

    def forward(self, x):
        return self.head(self.features(x))


class CNN_KAN(nn.Module):
    def __init__(self, hidden=64, degree=3):
        super().__init__()
        self.features = FeatureExtractor()
        self.head = ChebyKAN_Head(128, hidden, 10, degree)

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name):
    if name == 'mlp':
        return CNN_MLP()
    if name == 'kan':
        return CNN_KAN()
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            pred = out.argmax(1)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            correct += (pred == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    acc = correct / total
    f1, prec, rec, _ = _metrics(y_true, y_pred)
    return total_loss / total, acc, f1, prec, rec, y_true, y_pred


def _new_history():
    return {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_precision': [], 'val_recall': []}


def save_checkpoint(outdir, tag, model, epoch, best_acc, history):
    """Persist weights + history after every epoch, so a kill loses at most 1 epoch."""
    torch.save(model.state_dict(), os.path.join(outdir, f'last_{tag}.pth'))
    ckpt = {'epoch': epoch, 'best_acc': best_acc, 'history': history}
    with open(os.path.join(outdir, f'checkpoint_{tag}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(ckpt, f, ensure_ascii=False)


def load_checkpoint(outdir, tag, model, device):
    """Return (epoch, best_acc, history) if a checkpoint exists, else None."""
    ckpt_path = os.path.join(outdir, f'checkpoint_{tag}.json')
    last_path = os.path.join(outdir, f'last_{tag}.pth')
    if not (os.path.exists(ckpt_path) and os.path.exists(last_path)):
        return None
    with open(ckpt_path, encoding='utf-8') as f:
        ckpt = json.load(f)
    model.load_state_dict(torch.load(last_path, map_location=device,
                                     weights_only=True))
    return ckpt['epoch'], ckpt['best_acc'], ckpt['history']


def train(model, train_loader, test_loader, epochs, lr, device, outdir, tag,
          start_epoch=0, best_acc=0.0, history=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if history is None:
        history = _new_history()
    for _ in range(start_epoch):          # restore lr-schedule position on resume
        scheduler.step()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        run_loss, total, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            total += images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
        train_loss = run_loss / total
        train_acc = correct / total
        v_loss, v_acc, v_f1, v_prec, v_rec, _, _ = evaluate(model, test_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['val_f1'].append(v_f1)
        history['val_precision'].append(v_prec)
        history['val_recall'].append(v_rec)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(outdir, f'best_{tag}.pth'))

        save_checkpoint(outdir, tag, model, epoch + 1, best_acc, history)

        print(f'[{tag}] epoch {epoch+1}/{epochs} '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={v_loss:.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    print(f'[{tag}] BEST val_acc = {best_acc:.4f}')
    return history, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='both', choices=['mlp', 'kan', 'both'])
    ap.add_argument('--data_dir', default=r'C:\Users\秦宇欣\Desktop\2022-2026\stitp\kan\kan\k375训练(1)\k375训练\dataset')
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'results'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true',
                    help='resume from checkpoint_{tag}.json / last_{tag}.pth')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  dataset={args.data_dir}')

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch, args.workers, args.subset)
    print(f'train batches={len(train_loader)} test batches={len(test_loader)}')

    models = ['mlp', 'kan'] if args.model == 'both' else [args.model]
    summary = {}
    for name in models:
        model = build_model(name).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'\n=== {name.upper()} | params={n_params:,} ===')
        start_epoch, best_acc, history = 0, 0.0, None
        if args.resume:
            loaded = load_checkpoint(args.outdir, name, model, device)
            if loaded is not None:
                start_epoch, best_acc, history = loaded
                print(f'[resume] {name}: epoch={start_epoch} best_acc={best_acc:.4f}')
            else:
                print(f'[resume] {name}: no checkpoint found, starting fresh')
        history, best_acc = train(model, train_loader, test_loader, args.epochs,
                                  args.lr, device, args.outdir, name,
                                  start_epoch=start_epoch, best_acc=best_acc,
                                  history=history)
        summary[name] = {'best_val_acc': best_acc, 'n_params': n_params,
                         'history': history}

        # Confusion matrix on the test set with the best weights.
        model.load_state_dict(torch.load(os.path.join(args.outdir, f'best_{name}.pth'),
                                         weights_only=True))
        _, _, _, _, _, y_true, y_pred = evaluate(model, test_loader, device)
        _, _, _, cm = _metrics(y_true, y_pred)
        summary[name]['confusion_matrix'] = cm.tolist()
        summary[name]['class_names'] = CLASS_NAMES

    summary['_meta'] = {'seed': args.seed, 'epochs': args.epochs,
                        'batch': args.batch, 'lr': args.lr,
                        'subset': args.subset}
    with open(os.path.join(args.outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('\nDONE. summary saved to', os.path.join(args.outdir, 'summary.json'))


if __name__ == '__main__':
    main()
