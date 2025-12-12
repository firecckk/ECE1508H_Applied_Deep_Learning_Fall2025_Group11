# example usage: python evaluate.py --data-dir C:/Users/hardy/Desktop/OCT --pin-memory
import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import vig
import csv
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from tqdm import tqdm
from torchvision import models
from preprocess import build_dataloaders
from scipy.special import softmax

# helper to replace first conv2d to match in_channels
def replace_first_conv(model: nn.Module, in_channels: int):
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parts = full_name.split('.')
            parent = model

            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)

            last = parts[-1]

            # get the original conv object
            old_conv = getattr(parent, last) if not last.isdigit() else parent[int(last)]

            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None)
            )

            # init new conv weights from old_conv weights where possible
            with torch.no_grad():
                old_w = old_conv.weight.data # [out, in_old, k, k]
                in_old = old_w.shape[1]

                if in_old == in_channels:
                    new_conv.weight.data.copy_(old_w)
                else:
                    if in_old == 3 and in_channels == 1:
                        # average RGB -> single channel
                        new_conv.weight.data.copy_(old_w.sum(dim=1, keepdim=True))
                    elif in_old == 1 and in_channels == 3:
                        # duplicate single channel to RGB and scale
                        new_conv.weight.data.copy_(old_w.repeat(1, 3, 1, 1) / 3.0)
                    else:
                        mean = old_w.mean(dim=1, keepdim=True) # [out,1,k,k]
                        new_conv.weight.data.copy_(mean.repeat(1, in_channels, 1, 1))

                # copy bias
                if old_conv.bias is not None:
                    new_conv.bias.data.copy_(old_conv.bias.data)

            # attach new_conv into parent
            if last.isdigit():
                parent[int(last)] = new_conv
            else:
                setattr(parent, last, new_conv)

            return

# helpers - replace head
def replace_last_linear(model: nn.Module, num_classes: int) -> bool:
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            # find parent
            parts = name.split('.')
            parent = model

            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)

            last = parts[-1]
            new_linear = nn.Linear(module.in_features, num_classes)

            if last.isdigit():
                parent[int(last)] = new_linear
            else:
                setattr(parent, last, new_linear)

            return True
        
    return False

# builder for torchvision models
def build_torchvision_model(
    model_name: str,
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = False,
    img_size: int = 224
) -> Tuple[nn.Module, Any]:
    mname = model_name.lower()

    if not hasattr(models, mname):
        raise ValueError(f"torchvision.models has no {mname}")
    
    ctor = getattr(models, mname)

    # find pretrained weights
    weights_choice = None
    weights_name = None

    if pretrained:
        for attr in dir(models):
            low = attr.lower()

            if low.startswith(mname) and low.endswith('_weights'):
                weights_enum = getattr(models, attr)

                try:
                    weights_choice = getattr(weights_enum, 'DEFAULT', None) or (list(weights_enum)[0] if len(list(weights_enum)) > 0 else None)
                except Exception:
                    weights_choice = None

                if weights_choice is not None:
                    weights_name = f"{attr}.{getattr(weights_choice,'name',str(weights_choice))}"

                break
    try:
        model = ctor(weights=weights_choice if weights_choice is not None else None)
    except Exception:
        model = ctor()

    if hasattr(model, 'image_size'):
        try:
            model.image_size = img_size
        except Exception:
            pass

    if in_channels != 3:
        replace_first_conv(model, in_channels)

    # replace head
    replaced = replace_last_linear(model, num_classes)

    if not replaced:
        try:
            if hasattr(model, 'classifier'):
                model.classifier = nn.Linear(1024, num_classes)
            elif hasattr(model, 'fc'):
                model.fc = nn.Linear(getattr(model, 'fc').in_features, num_classes)
        except Exception:
            pass

    return model, weights_name

# load a .pth file
def load_saved_weights(model: nn.Module, path: str, device: torch.device):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"weights file not found: {path}")
    
    data = torch.load(str(p), map_location=device)

    if isinstance(data, dict):
        if 'model_state' in data:
            state = data['model_state']
        elif 'state_dict' in data:
            state = data['state_dict']
        else:
            keys = list(data.keys())

            if len(keys) > 0 and isinstance(keys[0], str) and '.' in keys[0]:
                state = data
            else:
                candidate = None

                for k in ['model', 'net', 'module']:
                    if k in data and isinstance(data[k], dict):
                        candidate = data[k]

                        break

                if candidate is not None:
                    state = candidate
                else:
                    try:
                        model.load_state_dict(data, strict=False)

                        return True
                    except Exception as e:
                        raise RuntimeError(f"unable to locate state_dict inside {path}: {e}")
    else:
        raise RuntimeError(f"expected a dict saved to {path}, got {type(data)}")

    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        new_state = {}

        for k, v in state.items():
            nk = k

            if k.startswith('module.'):
                nk = k[len('module.'):]

            new_state[nk] = v

        model.load_state_dict(new_state, strict=False)

    return True

# evaluation helper
def evaluate_model_on_loader(model: nn.Module, loader, device: torch.device):
    model.eval()

    all_logits = []
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="evaluating", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            outputs = model(imgs)

            if isinstance(outputs, tuple) or isinstance(outputs, list):
                outputs = outputs[0]

            logits = outputs.detach().cpu()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_logits.append(logits.numpy())
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_scores = np.vstack(all_logits) # [N, C]
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    return y_true, y_pred, y_scores

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, label_names: Dict[int,str]):
    num_classes = y_scores.shape[1]

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # per-class sensitivity (recall) and specificity
    per_class = {}
    N = cm.sum()

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = N - TP - FP - FN
        sens = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        spec = float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class[i] = {
            'label': label_names.get(i, str(i)),
            'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
            'sensitivity': sens,
            'specificity': spec,
            'support': support
        }

    # macro/weighted sensitivity/specificity
    sens_list = [per_class[i]['sensitivity'] for i in range(num_classes)]
    spec_list = [per_class[i]['specificity'] for i in range(num_classes)]
    macro_sens = float(np.mean(sens_list))
    macro_spec = float(np.mean(spec_list))

    # weighted average by support
    supports = np.array([per_class[i]['support'] for i in range(num_classes)], dtype=float)
    total = supports.sum() if supports.sum() > 0 else 1.0
    weighted_sens = float(np.sum(np.array(sens_list) * supports) / total)
    weighted_spec = float(np.sum(np.array(spec_list) * supports) / total)

    # weighted error: 1 - accuracy
    weighted_error = 1.0 - acc

    # AUC ROC (multiclass OVR weighted)
    try:
        probs = softmax(y_scores, axis=1)
    except Exception:
        probs = np.exp(y_scores) / np.exp(y_scores).sum(axis=1, keepdims=True)

    try:
        auc = float(roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'))
    except Exception:
        auc = float('nan')

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'macro_sensitivity': macro_sens,
        'macro_specificity': macro_spec,
        'weighted_sensitivity': weighted_sens,
        'weighted_specificity': weighted_spec,
        'weighted_error': weighted_error,
        'auc_ovr_weighted': auc,
        'n_samples': int(len(y_true))
    }

    return results, probs

# plotting helpers
def plot_metric_bar(models_names, metric_values, metric_label, out_path: Path):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(models_names))
    vals = np.array(metric_values)
    bars = plt.bar(x, vals, tick_label=models_names)
    plt.ylim(0.0, 1.0 if metric_label.lower() != 'error' else 1.0)
    plt.ylabel(metric_label)

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

    plt.title(f"Model comparison: {metric_label}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(cm, labels, out_path: Path, title="confusion matrix"):
    cm = np.array(cm)
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.Blues
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.xlabel('predicted')
    plt.ylabel('true')
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_model_rocs(models_probs, y_true, n_classes, labels_map, out_path: Path):
    plt.figure(figsize=(8, 8))

    for name, probs in models_probs.items():
        try:
            y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)

            for i, t in enumerate(y_true):
                y_true_bin[i, t] = 1
            
            auc = roc_auc_score(y_true, probs, multi_class='ovr', average='weighted')

            # compute micro ROC by flattening
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
            plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC curves (models comparison)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building dataloaders (using same preprocessing and seed=666)...")

    dataloaders_info = build_dataloaders({
        "data_dir": args.data_dir,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "val_ratio": 0.1,  # not needed here (we only want test split created by build_dataloaders),
        "test_ratio": args.test_ratio,
        "random_seed": args.seed,
        "num_workers": args.num_workers,
        "rgb": args.rgb,
        "shuffle_train": False,
        "pin_memory": args.pin_memory
    })
    test_loader = dataloaders_info['test_loader']
    idx_to_label = dataloaders_info['idx_to_label']
    mean = dataloaders_info['mean']
    std = dataloaders_info['std']
    cfg = dataloaders_info['config']

    print(f"num_workers used for evaluation dataloaders: {cfg['num_workers']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"evaluating on device: {device} (cuda available: {torch.cuda.is_available()})")

    # models list and default weight paths
    models_info = [
        {'name': 'convnext_tiny', 'builder': lambda: build_torchvision_model('convnext_tiny', num_classes=args.num_classes, in_channels=1, pretrained=False, img_size=args.img_size), 'weights': args.convnext_weights},
        {'name': 'densenet169', 'builder': lambda: build_torchvision_model('densenet169', num_classes=args.num_classes, in_channels=1, pretrained=False, img_size=args.img_size), 'weights': args.densenet_weights},
        {'name': 'resnet50', 'builder': lambda: build_torchvision_model('resnet50', num_classes=args.num_classes, in_channels=1, pretrained=False, img_size=args.img_size), 'weights': args.resnet_weights},
        {'name': 'swin_v2_t', 'builder': lambda: build_torchvision_model('swin_v2_t', num_classes=args.num_classes, in_channels=1, pretrained=False), 'weights': args.swin_weights},
        {'name': 'vit_b_16', 'builder': lambda: build_torchvision_model('vit_b_16', num_classes=args.num_classes, in_channels=1, pretrained=False, img_size=args.img_size), 'weights': args.vit_weights},
        {'name': 'vig_tiny', 'builder': lambda: vig.build_vig(variant='tiny', num_classes=args.num_classes, in_channels=1, img_size=args.img_size, k=args.vig_k, heads=args.vig_heads, drop_path_rate=args.vig_drop_path, head_dropout=args.vig_dropout)[0], 'weights': args.vig_weights}
    ]
    results = {}
    models_probs_for_roc = {}

    # iterate models
    for mi in models_info:
        name = mi['name']
        wpath = mi.get('weights')

        print(f"\n=== evaluating model: {name} ===")

        # build model
        try:
            model, pretrained_name = mi['builder']() if name != 'vig_tiny' else (mi['builder'](), None)
        except Exception as e:
            try:
                mtmp = mi['builder']()

                if isinstance(mtmp, tuple) and len(mtmp) == 2:
                    model, pretrained_name = mtmp
                else:
                    model, pretrained_name = mtmp, None
            except Exception as e2:
                print(f"failed to construct {name}: {e2}")

                continue

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  model parameters: {n_params:,}")

        # load weights if given
        if wpath:
            try:
                print(f"  loading weights from: {wpath}")

                load_saved_weights(model, wpath, device)

                print("  weights loaded.")
            except Exception as e:
                print(f"  failed to load weights for {name} from {wpath}: {e}")
        else:
            print("  no weights path provided; evaluating randomly-initialized model (probably poor performance).")

        # evaluate
        t0 = time.time()

        y_true, y_pred, y_scores = evaluate_model_on_loader(model, test_loader, device)

        t1 = time.time()

        elapsed = t1 - t0

        print(f"  evaluation time: {elapsed:.1f}s | samples: {len(y_true)}")

        # compute metrics
        label_names = {k: v for k, v in idx_to_label.items()}
        metrics, probs = compute_all_metrics(y_true, y_pred, y_scores, label_names)
        metrics['evaluation_time_s'] = elapsed
        metrics['num_parameters'] = n_params
        results[name] = metrics
        models_probs_for_roc[name] = probs

        # save
        model_out = out_dir / name
        model_out.mkdir(parents=True, exist_ok=True)
        (model_out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

        np.save(model_out / 'y_true.npy', y_true)
        np.save(model_out / 'y_pred.npy', y_pred)
        np.save(model_out / 'y_probs.npy', probs)

        # confusion matrix plot
        cm = metrics['confusion_matrix']
        labels = [label_names[i] for i in range(args.num_classes)]
        plot_confusion_matrix(cm, labels, model_out / 'confusion_matrix.png', title=f"confusion matrix ({name})")

        print(f"  saved metrics and confusion matrix to {model_out}")

    print("\n" + "="*80)
    print("SUMMARY: numerical results for each model")

    header = ["model", "accuracy", "precision", "recall", "f1", "auc_ovr_weighted", "w_sensitivity", "w_specificity", "weighted_error", "n_params"]
    
    print("{:<18s} {:>8s} {:>9s} {:>7s} {:>7s} {:>8s} {:>8s} {:>8s} {:>9s} {:>12s}".format(*header))

    for name, metrics in results.items():
        print("{:<18s} {:8.4f} {:9.4f} {:7.4f} {:7.4f} {:8.4f} {:8.4f} {:8.4f} {:9.4f} {:12,d}".format(
            name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics.get('auc_ovr_weighted', float('nan')),
            metrics.get('weighted_sensitivity', float('nan')),
            metrics.get('weighted_specificity', float('nan')),
            metrics.get('weighted_error', float('nan')),
            metrics.get('num_parameters', 0)
        ))

    # create comparison plots for each metric
    model_names = list(results.keys())
    accs = [results[m]['accuracy'] for m in model_names]
    precs = [results[m]['precision'] for m in model_names]
    recs = [results[m]['recall'] for m in model_names]
    f1s = [results[m]['f1'] for m in model_names]
    aucs = [results[m].get('auc_ovr_weighted', np.nan) for m in model_names]
    wsens = [results[m].get('weighted_sensitivity', np.nan) for m in model_names]
    wspecs = [results[m].get('weighted_specificity', np.nan) for m in model_names]
    werrs = [results[m].get('weighted_error', np.nan) for m in model_names]

    (out_dir / 'aggregated_results.json').write_text(json.dumps(results, indent=2))

    # plot bars
    plot_metric_bar(model_names, accs, 'accuracy', out_dir / 'metric_accuracy.png')
    plot_metric_bar(model_names, precs, 'precision', out_dir / 'metric_precision.png')
    plot_metric_bar(model_names, recs, 'recall', out_dir / 'metric_recall.png')
    plot_metric_bar(model_names, f1s, 'f1', out_dir / 'metric_f1.png')
    plot_metric_bar(model_names, aucs, 'auc_ovr_weighted', out_dir / 'metric_auc.png')
    plot_metric_bar(model_names, wsens, 'weighted_sensitivity', out_dir / 'metric_weighted_sensitivity.png')
    plot_metric_bar(model_names, wspecs, 'weighted_specificity', out_dir / 'metric_weighted_specificity.png')
    plot_metric_bar(model_names, werrs, 'weighted_error', out_dir / 'metric_weighted_error.png')

    # ROC curves
    plot_model_rocs(models_probs_for_roc, np.load(out_dir / model_names[0] / 'y_true.npy') if len(model_names) > 0 else np.array([]), args.num_classes, idx_to_label, out_dir / 'roc_comparison.png')

    csv_path = out_dir / 'results_table.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for name in model_names:
            m = results[name]
            writer.writerow([name, m['accuracy'], m['precision'], m['recall'], m['f1'], m.get('auc_ovr_weighted', ''), m.get('weighted_sensitivity', ''), m.get('weighted_specificity',''), m.get('weighted_error',''), m.get('num_parameters',0)])
    
    elapsed_total = time.time() - start_time
    h, rem = divmod(int(elapsed_total), 3600)
    m, s = divmod(rem, 60)

    print("\nEvaluation finished.")
    print(f"  outputs saved to: {out_dir}")
    print(f"  total time: {h}h {m}m {s}s")
    print("="*80)

# flags
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate multiple trained models on test set and produce plots/metrics")
    parser.add_argument('--data-dir', type=str, required=True, help='path to dataset root (same structure as used in preprocess)')
    parser.add_argument('--out-dir', type=str, default='./evaluation_outputs', help='where to save evaluation outputs')
    parser.add_argument('--convnext-weights', type=str, default='./outputs/Z/ConvNeXt_tiny/best_model.pth')
    parser.add_argument('--densenet-weights', type=str, default='./outputs/densenet_search/run_004_lr0.0005_wd0.01_bs16/best_model.pth')
    parser.add_argument('--resnet-weights', type=str, default='./outputs/resnet_search/run_003_lr0.0001_wd0.0001_bs64/best_model.pth')
    parser.add_argument('--swin-weights', type=str, default='./outputs/swin_search/run_003_lr0.0001_wd0.01_bs16/best_model.pth')
    parser.add_argument('--vit-weights', type=str, default='./outputs/Z/ViT_b_16/run_001_lr5e-05_wd1e-05_bs32/best_model.pth')
    parser.add_argument('--vig-weights', type=str, default='./outputs/vig_tiny_search/run_003_lr0.002_wd0.01_bs64_k9_h2_dp0.1_schedCosineAnnealingLR/best_model.pth')
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size used to compute mean/std and test loader - override if needed')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='test ratio used by preprocess (must match how you split earlier)')
    parser.add_argument('--num-workers', type=int, default=max(0, min(12, (os.cpu_count() or 4) - 1)))
    parser.add_argument('--rgb', action='store_true', help='dataset is RGB (default False)')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--vig-k', type=int, default=9)
    parser.add_argument('--vig-heads', type=int, default=2)
    parser.add_argument('--vig-dropout', type=float, default=0.2)
    parser.add_argument('--vig-drop-path', type=float, default=0.1)

    args = parser.parse_args()

    if not args.rgb:
        args.rgb = False

    main(args)