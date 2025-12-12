'''
train an isotropic vig classifier with hyperparameter search and resume support
example usage: python train_vig.py --data-dir C:/Users/hardy/Desktop/OCT --pin-memory --mixed-precision
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
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess import build_dataloaders
from vig import build_vig, apply_kaiming_init

# for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# number of learnable parameters
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# compute metrics
def compute_metrics(outputs, labels):
    preds = outputs.argmax(dim=1).cpu()
    labels = labels.cpu()
    acc = (preds == labels).float().mean().item()
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    return acc, prec, rec, f1

# train loop
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
            with torch.amp.autocast(device_type='cuda'):
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

# eval loop
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

# run one hyperparameter configuration
def run_experiment(cfg, hp, out_dir, device, resume=False, verbose=True):
    start_time = time.time()
    run_id = f"lr{hp['lr']}_wd{hp['weight_decay']}_bs{hp['batch_size']}_dropout{hp['head_dropout']}_k{hp['k']}_h{hp['heads']}_dp{hp['drop_path']}_sched{hp['scheduler']}"
    
    if verbose:
        print(f"\n--- starting run {run_id} ---")
        print(f"hyperparams: lr={hp['lr']} | wd={hp['weight_decay']} | bs={hp['batch_size']} | heads={hp['heads']} | k={hp['k']} | drop_path={hp['drop_path']} | head_dropout={hp['head_dropout']} | scheduler={hp['scheduler']}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume check
    summary_path = out_dir / 'summary.json'
    if resume and summary_path.exists():
        print(f"    found existing results for {run_id}, skipping due to --resume")
        try:
            return json.loads(summary_path.read_text()), None  # return existing summary
        except Exception:
            return None, None

    # build dataloaders
    dl_cfg = cfg.copy()
    dl_cfg['batch_size'] = int(hp['batch_size'])
    
    print(f"    building dataloaders (preprocessing) | batch_size={dl_cfg['batch_size']} | num_workers={dl_cfg.get('num_workers')}")
    
    out = build_dataloaders(dl_cfg)
    train_loader, val_loader, test_loader = out['train_loader'], out['val_loader'], out['test_loader']
    class_weights, idx_to_label = out['class_weights'], out['idx_to_label']
    num_classes = len(idx_to_label)
    in_channels = 3 if dl_cfg.get('rgb', False) else 1

    # build vig
    model, pretrained_weights_name = build_vig(
        variant=cfg.get('vig_variant', 'tiny'),
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=cfg['img_size'],
        k=hp['k'],
        heads=hp['heads'],
        drop_path_rate=hp['drop_path'],
        head_dropout=hp['head_dropout'],
        pretrained=cfg.get('pretrained', False)
    )
    device_str = str(device)
    model = model.to(device)

    # apply kaiming initialization
    if not cfg.get('pretrained', False):
        try:
            apply_kaiming_init(model)
        except Exception:
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, 'weight', None) is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # count parameters
    n_params = count_parameters(model)

    if verbose:
        print(f"    model variant: {cfg.get('vig_variant','tiny')} | params: {n_params:,} | device: {device_str}")

        if pretrained_weights_name:
            print(f"    pretrained weights: {pretrained_weights_name}")

    # training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=float(hp['lr']), weight_decay=float(hp['weight_decay']))
    
    # scheduler
    scheduler_choice = hp.get('scheduler','ReduceLROnPlateau')

    if scheduler_choice == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)

    scaler = torch.amp.GradScaler(enabled=(cfg['mixed_precision'] and torch.cuda.is_available()))

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

    # training loop with early stopping based on val f1
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg['mixed_precision']
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)

        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            try:
                scheduler.step(val_f1)
            except Exception:
                scheduler.step(-val_f1)
        else:
            try:
                scheduler.step()
            except Exception:
                pass

        # record
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

        # early stopping
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
        print(f"epoch {epoch:03d}/{epochs} | time: {epoch_t1 - epoch_t0:.1f}s | train loss: {train_loss:.4f} acc: {train_acc*100:.2f}% f1: {train_f1:.3f} | val loss: {val_loss:.4f} acc: {val_acc*100:.2f}% f1: {val_f1:.3f}")

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss
        }, out_dir / 'last_ckpt.pth')

        if patience_counter >= patience:
            print(f"early stopping (no val_f1 improvement for {patience} epochs). best val_f1={best_val_f1:.4f} at epoch {best_epoch}.")
            
            break

    # restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)

    torch.save({'model_state': model.state_dict(), 'hyperparameters': hp, 'best_epoch': best_epoch}, out_dir / 'best_model.pth')

    # save history
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

    # summary
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

    # plots
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

    return summary, history

# hyperparameter search
def main(args):
    set_seed(args.seed)

    cfg = {
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'num_workers': args.num_workers,
        'rgb': args.rgb,
        'random_seed': args.seed,
        'pin_memory': args.pin_memory,
        'vig_variant': args.vig,
        'pretrained': args.pretrained,
        'mixed_precision': args.mixed_precision,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'shuffle_train': True
    }

    print("\n=== preprocessing / data info ===")
    print(f"  data_dir: {cfg['data_dir']}")
    print(f"  img_size: {cfg['img_size']}")
    print(f"  default batch_size (informational): {args.batch_size}")
    print(f"  num_workers: {cfg['num_workers']}")
    print(f"  pin_memory: {cfg['pin_memory']}")
    print("=================================\n")

    # informational preprocessing
    tmp_cfg = cfg.copy()
    tmp_cfg['batch_size'] = args.batch_size
    print("building initial dataloaders (informational run) - running preprocessing now...")
    out_info = build_dataloaders(tmp_cfg)
    try:
        train_loader = out_info.get('train_loader')
        val_loader = out_info.get('val_loader')
        test_loader = out_info.get('test_loader')
        idx_to_label = out_info.get('idx_to_label')
        class_weights = out_info.get('class_weights')
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
    dropout_list = [float(x) for x in args.dropout_list.split(',')]
    k_list = [int(x) for x in args.k_list.split(',')]
    h_list = [int(x) for x in args.h_list.split(',')]
    scheduler_list = [x for x in args.scheduler_list.split(',')]
    drop_path_list = [float(x) for x in args.drop_path_list.split(',')]

    # build combos
    combos = []
    if args.search_method == 'grid':
        for lr, wd, bs, hd, k, h, sch, dp in itertools.product(lr_list, wd_list, bs_list, dropout_list, k_list, h_list, scheduler_list, drop_path_list):
            combos.append({
                'lr': lr, 'weight_decay': wd, 'batch_size': bs,
                'head_dropout': hd, 'k': k, 'heads': h, 'scheduler': sch, 'drop_path': dp
            })
    else:
        random.seed(args.seed)

        for _ in range(args.num_trials):
            combos.append({
                'lr': random.choice(lr_list),
                'weight_decay': random.choice(wd_list),
                'batch_size': random.choice(bs_list),
                'head_dropout': random.choice(dropout_list),
                'k': random.choice(k_list),
                'heads': random.choice(h_list),
                'scheduler': random.choice(scheduler_list),
                'drop_path': random.choice(drop_path_list)
            })

    root_out = Path(args.save_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2

            print(f"using device: {gpu_name} | memory: {mem_alloc:.1f} MB / {mem_total:.1f} MB")
        except Exception:
            print(f"using device: {device}")
    else:
        print(f"using device: {device}")

    all_summaries = []
    best_summary = None

    total_runs = len(combos)
    print(f"\nstarting hyperparameter search: method={args.search_method} | total runs={total_runs}")

    # iterate combos
    for i, hp in enumerate(combos, start=1):
        run_name = f"run_{i:03d}_lr{hp['lr']}_wd{hp['weight_decay']}_bs{hp['batch_size']}_k{hp['k']}_h{hp['heads']}_dp{hp['drop_path']}_sched{hp['scheduler']}"
        run_out = root_out / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{total_runs}] starting trial: {run_name}")
        print(f"  hyperparams: lr={hp['lr']} | wd={hp['weight_decay']} | bs={hp['batch_size']} | k={hp['k']} | heads={hp['heads']} | drop_path={hp['drop_path']} | head_dropout={hp['head_dropout']} | scheduler={hp['scheduler']}")
        print(f"  saving outputs to: {run_out}")

        # resume check
        # if summary exists, skip
        summary, history = run_experiment(cfg, hp, run_out, device, resume=args.resume, verbose=True)

        if summary is None:
            print(f"  skipping {run_name} due to resume/read error.")

            continue

        all_summaries.append(summary)

        # update best based on validation F1
        if best_summary is None or summary['best_val_f1'] > best_summary['best_val_f1']:
            best_summary = summary

            print(f"  new best combo: mean val f1={best_summary['best_val_f1']:.4f} -> {run_name}")

    aggregated = {'runs': all_summaries, 'best_run': best_summary}
    (root_out / 'aggregated_results.json').write_text(json.dumps(aggregated, indent=2))

    # final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (best run by validation F1):")

    if best_summary:
        best = best_summary
        hp = best.get('hyperparameters', {})

        print(f"  run id         : {best.get('run_id')}")
        print(f"  hyperparams    : lr={hp.get('lr')} | weight_decay={hp.get('weight_decay')} | batch_size={hp.get('batch_size')} | k={hp.get('k')} | heads={hp.get('heads')} | drop_path={hp.get('drop_path')}")
        print(f"  best val F1    : {best.get('best_val_f1'):.4f} at epoch {best.get('best_epoch')}")
        print(f"  test metrics   : loss={best.get('test_loss'):.4f} acc={best.get('test_acc'):.4f} prec={best.get('test_prec'):.4f} rec={best.get('test_rec'):.4f} f1={best.get('test_f1'):.4f}")
        print(f"  num parameters : {best.get('num_parameters'):,}")
        print(f"  device         : {best.get('device')}")
        print(f"  pretrained used: {best.get('pretrained_weights')}")
        print(f"  run time (s)   : {best.get('run_time_seconds'):.1f}")
        print(f"  results folder : {root_out / best.get('run_id')}")
    else:
        print("  no completed runs found.")

    print("=" * 80 + "\n")
    print("all run summaries saved to:", str(root_out / 'aggregated_results.json'))
    print("done.")

# flags
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train isotropic vig on oct dataset (with hyperparam search)')
    parser.add_argument('--data-dir', type=str, required=True, help='dataset path')
    parser.add_argument('--vig', type=str, default='tiny', help='vig variant (tiny/small/base) (default tiny)')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained weights if available (vig default: train from scratch)')
    parser.add_argument('--rgb', action='store_true', help='use rgb images (default grayscale)')
    parser.add_argument('--img-size', type=int, default=224, help='image size (square). ensure divisible by 16 for overlap stem')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs (default 150)')
    parser.add_argument('--batch-size', type=int, default=32, help='default batch size (informational)')
    parser.add_argument('--lr', type=float, default=1e-3, help='default lr (not used when searching)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='default wd (not used when searching)')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=max(0, min(12, (os.cpu_count() or 4) - 1)), help='number of dataloader workers (printed)')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--save-dir', type=str, default='./outputs/vig_tiny_search')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true', help='use amp (fp16) if cuda available')
    parser.add_argument('--search-method', type=str, choices=['random', 'grid'], default='random', help='grid or random search (default random)')
    parser.add_argument('--lr-list', type=str, default="2e-3,1e-3,5e-4,1e-4", help='comma-separated learning rate candidates (default 2e-3,1e-3,5e-4,1e-4)')
    parser.add_argument('--wd-list', type=str, default="0.05,0.01,0.001", help='comma-separated weight decay candidates (default 0.05,0.01,0.001)')
    parser.add_argument('--bs-list', type=str, default="16,32,64", help='comma-separated batch size candidates (default 16,32,64)')
    parser.add_argument('--dropout-list', type=str, default="0.2,0.4", help='comma-separated head dropout candidates (default 0.2,0.4)')
    parser.add_argument('--k-list', type=str, default="6,9,12,15", help='comma-separated k (neighbors) candidates (default 6,9,12,15)')
    parser.add_argument('--h-list', type=str, default="1,2,4", help='comma-separated heads candidates (default 1,2,4)')
    parser.add_argument('--scheduler-list', type=str, default="ReduceLROnPlateau,CosineAnnealingLR", help='comma-separated scheduler choices (default ReduceLROnPlateau,CosineAnnealingLR)')
    parser.add_argument('--drop-path-list', type=str, default="0.0,0.1", help='comma-separated drop_path candidates (default 0.0,0.1)')
    parser.add_argument('--num-trials', type=int, default=3, help='number of random trials (for random search) (default 5)')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='early stopping patience (val F1) (default 15)')
    parser.add_argument('--resume', action='store_true', help='if set, skip combos that already have results on disk')
    
    args = parser.parse_args()
    main(args)