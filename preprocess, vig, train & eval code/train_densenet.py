'''
train a densenet classifier (default: densenet169)
- example usage: python train_densenet.py --data-dir C:/Users/hardy/Desktop/OCT --pin-memory --mixed-precision
'''
import os
import time
import json
import csv
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess import build_dataloaders

# for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# number of learnable parameters
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# kaiming initialization if not pretrained
def apply_kaiming_init(model: nn.Module):
    for m in model.modules():
        # conv layers
        if isinstance(m, nn.Conv2d):
            if getattr(m, 'weight', None) is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

        # linear layers
        elif isinstance(m, nn.Linear):
            if getattr(m, 'weight', None) is not None:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5) if hasattr(math, 'sqrt') else 1.0)

            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

# adapt input channels
def replace_first_conv3(model: nn.Module, in_channels: int, pretrained: bool):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and getattr(module, 'in_channels', None) == 3:
            parts = name.split('.')
            parent = model

            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)

            last = parts[-1]
            conv = getattr(parent, last)
            new_conv = nn.Conv2d(
                in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=(conv.bias is not None)
            )

            if pretrained:
                with torch.no_grad():
                    w = conv.weight.data

                    if in_channels == 1:
                        new_conv.weight.data[:] = w.mean(dim=1, keepdim=True)
                    else:
                        repeat_factor = (in_channels + 2) // 3
                        rep = w.repeat(1, repeat_factor, 1, 1)[:, :in_channels, :, :].clone()
                        rep.mul_(3.0 / float(in_channels))
                        new_conv.weight.data[:] = rep

                    if conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data[:] = conv.bias.data

            if last.isdigit():
                parent[int(last)] = new_conv
            else:
                setattr(parent, last, new_conv)

            return True
        
    return False

# returns (model, pretrained_weights_name_or_None)
def build_model(model_name: str, num_classes: int = 4, in_channels: int = 1, pretrained: bool = False):
    model_name = model_name.lower()

    if not hasattr(models, model_name):
        raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

    ctor = getattr(models, model_name)

    # find weights enum if pretrained
    weights_choice = None
    weights_name = None

    if pretrained:
        for attr in dir(models):
            low = attr.lower()

            if low.startswith(model_name) and low.endswith('_weights'):
                weights_enum = getattr(models, attr)

                try:
                    # pick first available weight option
                    weights_choice = list(weights_enum)[0]
                    weights_name = f"{attr}.{weights_choice.name}"
                except Exception:
                    weights_choice = None

                break

    # instantiate model
    model = ctor(weights=weights_choice if pretrained else None)

    # adapt input channels
    if in_channels != 3:
        replaced = replace_first_conv3(model, in_channels=in_channels, pretrained=bool(pretrained))
        
        if not replaced:
            try:
                if hasattr(model, 'features') and hasattr(model.features, 'conv0'):
                    conv = model.features.conv0
                    new_conv = nn.Conv2d(
                        in_channels, conv.out_channels,
                        kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None)
                    )

                    if pretrained:
                        with torch.no_grad():
                            w = conv.weight.data

                            if in_channels == 1:
                                new_conv.weight.data[:] = w.mean(dim=1, keepdim=True)
                            else:
                                repeat_factor = (in_channels + 2)//3
                                rep = w.repeat(1, repeat_factor, 1, 1)[:, :in_channels, :, :].clone()
                                rep.mul_(3.0 / float(in_channels))
                                new_conv.weight.data[:] = rep

                            if conv.bias is not None and new_conv.bias is not None:
                                new_conv.bias.data[:] = conv.bias.data

                    model.features.conv0 = new_conv
                    replaced = True

            except Exception:
                pass

    # replace classifier head
    replaced = False

    if hasattr(model, 'classifier') and isinstance(getattr(model, 'classifier'), nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        replaced = True

    # fallback
    if not replaced:
        for name in ['head', 'fc', 'classifier']:
            if hasattr(model, name):
                attr = getattr(model, name)

                if isinstance(attr, nn.Linear):
                    in_features = attr.in_features
                    setattr(model, name, nn.Linear(in_features, num_classes))
                    replaced = True

                    break

    if not replaced:
        try:
            model.classifier = nn.Linear(512, num_classes)
        except Exception:
            pass

    return model, (weights_name if pretrained else None)

# compute accuracy, precision, recall, f1
def compute_metrics(outputs, labels):
    preds = outputs.argmax(dim=1).cpu()
    labels = labels.cpu()
    acc = (preds == labels).float().mean().item()
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    return acc, prec, rec, f1

# train for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()

    total_loss = 0.0
    all_outputs = []
    all_labels = []
    pbar = tqdm(loader, desc='train', leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        all_outputs.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    outputs = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(outputs, labels)

    return avg_loss, acc, prec, rec, f1

# evaluate for validation/test
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='eval', leave=False)

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * imgs.size(0)
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    outputs = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(outputs, labels)

    return avg_loss, acc, prec, rec, f1

import math  # used by kaiming fallback


# run single hyperparameter configuration
def run_experiment(cfg, hp, out_dir, device, verbose=True):
    start_time = time.time()
    run_id = f"lr{hp['lr']}_wd{hp['weight_decay']}_bs{hp['batch_size']}_seed{cfg['random_seed']}"

    if verbose:
        print(f"\n--- starting run {run_id} ---")
        print(f"hyperparameters: lr={hp['lr']} | weight_decay={hp['weight_decay']} | batch_size={hp['batch_size']}")

    # dataloaders for this batch_size
    dataloader_cfg = cfg.copy()
    dataloader_cfg['batch_size'] = int(hp['batch_size'])

    print(f"building dataloaders for batch_size={dataloader_cfg['batch_size']} | num_workers={dataloader_cfg.get('num_workers')} ...")
    
    out = build_dataloaders(dataloader_cfg)

    train_loader, val_loader, test_loader = out['train_loader'], out['val_loader'], out['test_loader']
    class_weights, idx_to_label = out['class_weights'], out['idx_to_label']
    num_classes = len(idx_to_label)
    in_channels = 3 if dataloader_cfg.get('rgb', False) else 1

    # build model
    model, pretrained_weights_name = build_model(
        cfg['model_name'], num_classes=num_classes,
        in_channels=in_channels,
        pretrained=cfg['pretrained']
    )
    device_str = str(device)
    model = model.to(device)

    # if not pretrained, apply kaiming initialization
    if not cfg['pretrained']:
        apply_kaiming_init(model)

    # count parameters
    n_params = count_parameters(model)

    if verbose:
        print(f"model: {cfg['model_name']} | params: {n_params:,} | device: {device_str}")

        if pretrained_weights_name:
            print(f"pretrained weights used: {pretrained_weights_name}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg['mixed_precision'] and torch.cuda.is_available()))

    epochs = cfg['epochs']
    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    patience = cfg['early_stopping_patience']

    history = {
        'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []
    }

    # train loop with early stopping on val_f1
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg['mixed_precision']
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)

        # scheduler step using val_f1
        try:
            scheduler.step(val_f1)
        except Exception:
            try:
                scheduler.step(-val_f1)
            except Exception:
                pass

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_prec'].append(train_prec)
        history['train_rec'].append(train_rec)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_f1'].append(val_f1)

        # early stopping on val_f1
        improved = val_f1 > best_val_f1 + 1e-8

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'val_loss': val_loss
            }
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_t1 = time.time()
        epoch_time = epoch_t1 - epoch_t0

        print(f"epoch {epoch:03d}/{epochs} | time: {epoch_time:.1f}s | "
              f"train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% f1: {train_f1:.3f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc*100:.2f}% f1: {val_f1:.3f}")

        # save last checkpoint
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            },
            out_dir / 'last_ckpt.pth'
        )

        # early stop
        if patience_counter >= patience:
            print(f"early stopping triggered (no val_f1 improvement for {patience} epochs). best val_f1={best_val_f1:.4f} at epoch {best_epoch}.")
            
            break

    # restore best state
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])

    # evaluate on test
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)

    # save best model and metadata
    best_model_path = out_dir / 'best_model.pth'
    torch.save({'model_state': model.state_dict(), 'hyperparameters': hp, 'best_epoch': best_epoch}, best_model_path)

    csv_path = out_dir / 'history.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['epoch',
                  'train_loss', 'train_acc', 'train_prec', 'train_rec', 'train_f1',
                  'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1']
        writer.writerow(header)

        for e in range(len(history['train_loss'])):
            row = [e+1,
                   history['train_loss'][e], history['train_acc'][e], history['train_prec'][e], history['train_rec'][e], history['train_f1'][e],
                   history['val_loss'][e], history['val_acc'][e], history['val_prec'][e], history['val_rec'][e], history['val_f1'][e]]
            writer.writerow(row)

    end_time = time.time()
    total_time = end_time - start_time
    summary = {
        'run_id': run_id,
        'hyperparameters': hp,
        'num_parameters': n_params,
        'device': device_str,
        'best_val_f1': float(best_val_f1),
        'best_epoch': int(best_epoch),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'test_prec': float(test_prec),
        'test_rec': float(test_rec),
        'test_f1': float(test_f1),
        'pretrained_weights': pretrained_weights_name,
        'run_time_seconds': total_time
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    # plot metrics vs epochs and include test as horizontal line
    epochs_range = list(range(1, len(history['train_loss']) + 1))
    def save_plot_with_test(y_train, y_val, test_val, ylabel, fname):
        plt.figure()
        plt.plot(epochs_range, y_train, label='train')
        plt.plot(epochs_range, y_val, label='val')

        if test_val is not None:
            plt.hlines(test_val, epochs_range[0], epochs_range[-1], colors='k', linestyles='--', label=f'test={test_val:.4f}')
        
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    save_plot_with_test(history['train_loss'], history['val_loss'], test_loss, 'loss', 'loss.png')
    save_plot_with_test(history['train_acc'], history['val_acc'], test_acc, 'accuracy', 'accuracy.png')
    save_plot_with_test(history['train_f1'], history['val_f1'], test_f1, 'f1', 'f1.png')
    save_plot_with_test(history['train_prec'], history['val_prec'], test_prec, 'precision', 'precision.png')
    save_plot_with_test(history['train_rec'], history['val_rec'], test_rec, 'recall', 'recall.png')

    if verbose:
        print(f"run {run_id} finished. best val_f1={best_val_f1:.4f} at epoch {best_epoch}. test f1={test_f1:.4f}. results saved to {out_dir}")

    return summary

# hyperparameter search
def main(args):
    # reproducibility
    set_seed(args.seed)

    base_cfg = {
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'num_workers': args.num_workers,
        'rgb': args.rgb,
        'random_seed': args.seed,
        'pin_memory': args.pin_memory,
        'model_name': args.model,
        'pretrained': args.pretrained,
        'mixed_precision': args.mixed_precision,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience
    }

    # preprocessing info (informational)
    print("\n=== preprocessing / data info ===")
    print(f"  data_dir: {base_cfg['data_dir']}")
    print(f"  img_size: {base_cfg['img_size']}")
    print(f"  default batch_size: {args.batch_size} (may be overridden per trial)")
    print(f"  num_workers: {base_cfg['num_workers']}")
    print(f"  pin_memory: {base_cfg['pin_memory']}")
    print("=================================\n")

    tmp_cfg = base_cfg.copy()
    tmp_cfg['batch_size'] = args.batch_size
    print("building initial dataloaders (informational run)...")
    out_info = build_dataloaders(tmp_cfg)

    if isinstance(out_info, dict):
        train_loader = out_info.get('train_loader')
        val_loader = out_info.get('val_loader')
        test_loader = out_info.get('test_loader')
        idx_to_label = out_info.get('idx_to_label')
        class_weights = out_info.get('class_weights')
        try:
            total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
            
            print(f"\ndataset size: {total} images")
            print(f"-> train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}")
            
            if idx_to_label:
                print(f"number of classes: {len(idx_to_label)}")
            
            if class_weights is not None:
                print(f"class weights (sample): {list(class_weights[:min(10,len(class_weights))])}")
        except Exception:
            pass

    # parse hyperparameter candidate lists
    lr_list = [float(x) for x in args.lr_list.split(',')]
    wd_list = [float(x) for x in args.wd_list.split(',')]
    bs_list = [int(x) for x in args.bs_list.split(',')]

    # build combos
    combos = []
    if args.search_method == 'grid':
        combos = [{'lr': lr, 'weight_decay': wd, 'batch_size': bs} for lr, wd, bs in itertools.product(lr_list, wd_list, bs_list)]
    else: # random search
        random.seed(args.seed)
        for _ in range(args.num_trials):
            combos.append({
                'lr': random.choice(lr_list),
                'weight_decay': random.choice(wd_list),
                'batch_size': random.choice(bs_list)
            })

    root_out = Path(args.save_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    best_summary = None
    total_runs = len(combos)
    print(f"\nstarting hyperparameter search: method={args.search_method} | total runs={total_runs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = str(device)
    if device.type == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"using device: {gpu_name} | memory: {mem_alloc:.1f} MB / {mem_total:.1f} MB")
        except Exception:
            print(f"using device: {device_str}")
    else:
        print(f"using device: {device_str}")

    # hyperparameter combos
    for i, hp in enumerate(combos, start=1):
        run_name = f"run_{i:03d}_lr{hp['lr']}_wd{hp['weight_decay']}_bs{hp['batch_size']}"
        run_out = root_out / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        # per-run config
        cfg = base_cfg.copy()
        cfg['batch_size'] = int(hp['batch_size'])
        cfg['model_name'] = args.model
        cfg['pretrained'] = args.pretrained

        print(f"\n[{i}/{total_runs}] starting trial: {run_name}")
        print(f"  hyperparams: lr={hp['lr']} | weight_decay={hp['weight_decay']} | batch_size={hp['batch_size']}")
        print(f"  saving outputs to: {run_out}")

        # run experiment
        summary = run_experiment(cfg, hp, run_out, device, verbose=True)
        all_summaries.append(summary)

        # choose best by highest validation f1
        if best_summary is None or summary['best_val_f1'] > best_summary['best_val_f1']:
            best_summary = summary

    aggregated_path = root_out / 'aggregated_results.json'
    aggregated = {'runs': all_summaries, 'best_run': best_summary}
    aggregated_path.write_text(json.dumps(aggregated, indent=2))

    print("\n" + "=" * 80)
    
    print("FINAL SUMMARY (best run by validation F1):")
    if best_summary:
        
        best = best_summary
        
        print(f"  run id         : {best.get('run_id')}")
        
        hp = best.get('hyperparameters', {})
        
        print(f"  hyperparams    : lr={hp.get('lr')} | weight_decay={hp.get('weight_decay')} | batch_size={hp.get('batch_size')}")
        print(f"  best val f1    : {best.get('best_val_f1'):.4f} at epoch {best.get('best_epoch')}")
        print(f"  test metrics   : loss={best.get('test_loss'):.4f} acc={best.get('test_acc'):.4f} prec={best.get('test_prec'):.4f} rec={best.get('test_rec'):.4f} f1={best.get('test_f1'):.4f}")
        print(f"  num parameters : {best.get('num_parameters'):,}")
        print(f"  device         : {best.get('device')}")
        print(f"  pretrained used: {best.get('pretrained_weights')}")
        print(f"  run time (s)   : {best.get('run_time_seconds'):.1f}")
        print(f"  results folder : {root_out / best.get('run_id')}")

    else:
        print("  no completed runs found.")
        
    print("=" * 80 + "\n")
    print("all run summaries saved to:", str(aggregated_path))
    print("done.")

# commands/flags
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model on oct dataset with tuning (densenet default)')
    parser.add_argument('--data-dir', type=str, required=True, help='dataset path')
    parser.add_argument('--model', type=str, default='densenet169', help='model name from torchvision.models (default densenet169)')
    parser.add_argument('--pretrained', action='store_true', help='use imagenet pretrained weights if available')
    parser.add_argument('--rgb', action='store_true', help='use rgb images (default is grayscale)')
    parser.add_argument('--img-size', type=int, default=224, help='image size (set 224/384 appropriately)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='default batch size (informational)')
    parser.add_argument('--lr', type=float, default=1e-4, help='default learning rate (not used when searching)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='default weight decay (not used when searching)')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=max(0, min(12, (os.cpu_count() or 4) - 1)), help='number of dataloader workers')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--save-dir', type=str, default='./outputs/densenet_search')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true', help='use amp (fp16) if cuda available')
    parser.add_argument('--search-method', type=str, choices=['random', 'grid'], default='random')
    parser.add_argument('--lr-list', type=str, default="1e-3,5e-4,1e-4", help='comma-separated learning rate candidates')
    parser.add_argument('--wd-list', type=str, default="1e-2,1e-3,1e-4", help='comma-separated weight-decay candidates')
    parser.add_argument('--bs-list', type=str, default="16,32,64", help='comma-separated batch size candidates')
    parser.add_argument('--num-trials', type=int, default=6, help='number of random trials (for random search)')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='early stopping patience (val f1)')
    args = parser.parse_args()
    main(args)