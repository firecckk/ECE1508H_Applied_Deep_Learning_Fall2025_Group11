'''
data preprocessing for OCT classification
'''
import os
import time
import random
import numpy as np
import torch
from collections import Counter
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
default configuration & hyperparameters
- can be overridden by build_dataloaders(config)
- NOTE: Change data_dir to PATH_TO_YOUR_DATASET
'''
DEFAULT_CONFIG = {
    "data_dir": r"C:/Users/hardy/Desktop/OCT", # path to dataset root
    "img_size": 224,
    "batch_size": 32,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
    "random_seed": 666,
    "num_workers": max(0, min(12, (os.cpu_count() or 4) - 1)),
    "rgb": False, # false for greyscale grayscale
    "shuffle_train": True,
    "pin_memory": True,
    "preview_out": "outputs/preview.png" # where the preview image will be saved
}

'''
dataset class
- reads image files and extracts the label from the filename
- label is extracted from filename prefix before the first '-' (e.g., 'CNV-001.jpg' -> 'CNV')
'''
class OCTDataset(Dataset):
    def __init__(self, img_dir: str, rgb: bool = False):
        self.img_dir = img_dir
        self.rgb = rgb
        self.image_paths: List[str] = []
        self.labels: List[str] = []

        # search directory and collect image paths and labels
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    path = os.path.join(root, f)
                    label = f.split('-')[0].upper()
                    self.image_paths.append(path)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")

        # assign 0, 1, 2, 3 to string labels alphabetically
        unique = sorted(set(self.labels))
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(unique)}
        self.idx_to_label: Dict[int, str] = {v: k for k, v in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[l] for l in self.labels]

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):
        # load and convert image to requested mode
        path = self.image_paths[idx]
        mode = "RGB" if self.rgb else "L"
        image = Image.open(path).convert(mode)
        label = self.labels[idx]

        return image, label

'''
resize & pad images while preserving the aspect ratio
- resize images proportionally so the longer side = size
- then pad to (size, size)
'''
def resize_and_pad_pil(img: Image.Image, size: int = 224, fill: int = 0) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    return ImageOps.expand(
        img_resized,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=fill
    )

'''
wrapper for resize_and_pad_pil()
'''
class ResizePadTransform:
    def __init__(self, size: int = 224, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:

        return resize_and_pad_pil(
            img,
            size=self.size,
            fill=self.fill
        )

'''
wrapper Dataset that applies a transform to an existing Subset
'''
class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):

        return len(self.subset)

    def __getitem__(self, idx):
        # retrieve original PIL image and label
        img, label = self.subset.dataset[self.subset.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

'''
get channel-wise mean & std across a loader
- expects tensors in [0,1]
'''
def compute_mean_std(loader: DataLoader, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
    cnt_pixels = 0
    sum_ = None
    sum_sq = None

    print("Computing train mean/std (this may take a while)...")

    for imgs, _ in tqdm(loader, desc="Mean/Std", leave=False):
        imgs = imgs.to(device)
        b, c, h, w = imgs.shape
        pixels = b * h * w

        if sum_ is None:
            sum_ = imgs.sum(dim=[0, 2, 3])
            sum_sq = (imgs ** 2).sum(dim=[0, 2, 3])
        else:
            sum_ += imgs.sum(dim=[0, 2, 3])
            sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])

        cnt_pixels += pixels

    mean = (sum_ / cnt_pixels)
    var = (sum_sq / cnt_pixels) - (mean ** 2)
    std = torch.sqrt(var.clamp(min=1e-9))

    return mean.cpu(), std.cpu()

'''
unnormalize a tensor image for visualization
- tensor [C,H,W] normalized -> numpy HxWxC
'''
def unnormalize_tensor_img(tensor_img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    mean = mean[:, None, None]
    std = std[:, None, None]
    un = tensor_img * std + mean
    arr = un.cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)

    return arr

'''
build train/val/test DataLoaders and return metadata
'''
def build_dataloaders(config: dict = None):
    # merge defaults with user config
    cfg = DEFAULT_CONFIG.copy()

    if config:
        cfg.update(config)

    # fix random seeds for reproducibility
    torch.manual_seed(cfg["random_seed"])
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    # timer
    t_start = time.perf_counter()

    print("\n=== Preprocessing configuration ===")

    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print("===================================\n")

    # walk directory
    print("Scanning dataset directory and building index...")

    dataset = OCTDataset(cfg["data_dir"], rgb=cfg["rgb"])
    total_images = len(dataset)

    print(f"Found {total_images} images across {len(dataset.label_to_idx)} classes.\n")

    # split the dataset
    indices = list(range(len(dataset)))
    labels = dataset.labels
    test_size = cfg["test_ratio"]

    print(f"Splitting test set (test_ratio={test_size})...")

    train_and_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=cfg["random_seed"]
    )

    val_fraction_total = cfg["val_ratio"]

    if val_fraction_total < 0 or val_fraction_total >= 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    
    remaining_fraction = 1.0 - test_size

    if remaining_fraction <= 0:
        raise ValueError("test_ratio must be less than 1.0.")
    
    val_frac_of_remaining = val_fraction_total / remaining_fraction
    val_frac_of_remaining = min(max(val_frac_of_remaining, 0.0), 1.0)

    print(
        f"Splitting validation set from remaining data so that val_fraction_of_total={val_fraction_total} "
        f"(val fraction of remaining={val_frac_of_remaining:.4f})..."
    )
    
    train_idx, val_idx = train_test_split(
        train_and_val_idx,
        test_size=val_frac_of_remaining,
        stratify=[labels[i] for i in train_and_val_idx],
        random_state=cfg["random_seed"]
    )

    print(f"Dataset size: {total_images} images")
    print(f"-> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_class_counts = Counter([dataset.labels[i] for i in train_idx])

    print("Train class distribution:", dict(train_class_counts))

    # class weights = normalized inverse frequency
    num_classes = len(dataset.label_to_idx)
    weights = torch.tensor(
        [1.0 / train_class_counts.get(i, 1.0) for i in range(num_classes)],
        dtype=torch.float
    )
    weights = weights / weights.sum() * num_classes
    torch.save(weights, "class_weights.pt")

    print("Class weights (saved to class_weights.pt):", weights.tolist())

    # resize & pad
    resize_pad = ResizePadTransform(size=cfg["img_size"], fill=0)

    # augmentations for training:
    # - RandomFlip
    # - RandomRotation
    # - RandomAffine: translate, scale and shear
    # - RandomResizedCrop: acts as zoom-in/zoom-out
    # - ColorJitter / Brightness/Contrast: intensity changes
    # - RandomHorizontalFlip
    train_augmentations = [
        transforms.RandomResizedCrop(cfg["img_size"], scale=(0.9, 1.1), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-8, 8)),
    ]

    # intensity augmentations
    if cfg["rgb"]:
        train_augmentations.append(transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.05, hue=0.02))
    else:
        train_augmentations.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.15)], p=0.4))

    base_train_transforms = [
        resize_pad,
    ]
    base_train_transforms += train_augmentations
    base_train_transforms += [
        transforms.ToTensor()
    ]

    # validation/test transforms: resize+pad -> ToTensor
    base_val_transforms = [
        resize_pad,
        transforms.ToTensor()
    ]

    # compute mean/std
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    stats_transform = transforms.Compose([resize_pad, transforms.ToTensor()])
    tmp_train_for_stats = TransformSubset(train_subset, transform=stats_transform)
    tmp_loader = DataLoader(
        tmp_train_for_stats,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = compute_mean_std(tmp_loader, device=device)
    mean_list = mean.tolist()
    std_list = std.tolist()

    print(f"Computed train mean: {mean_list}")
    print(f"Computed train std:  {std_list}")

    # final transforms
    normalize = transforms.Normalize(mean=mean_list, std=std_list)

    train_transform = transforms.Compose(base_train_transforms + [normalize])
    val_transform = transforms.Compose(base_val_transforms + [normalize])
    test_transform = val_transform

    # wrap subsets with transforms
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)
    test_dataset = TransformSubset(test_subset, transform=test_transform)

    # build DataLoaders
    print("\nBuilding DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle_train"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"]
    )

    # summary
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)

    print("\n" + "=" * 60)
    print("Preprocessing completed")
    print(f"Time elapsed: {h:d}h {m:02d}m {s:02d}s")
    print(f"Dataset: {total_images} images -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Image size (after resize/pad): {cfg['img_size']} x {cfg['img_size']}")
    print(f"Channels: {'RGB (3)' if cfg['rgb'] else 'Grayscale (1)'}")
    print(f"Batch size: {cfg['batch_size']} | Num workers: {cfg['num_workers']} | Pin memory: {cfg['pin_memory']}")
    print(f"Train class distribution: {dict(train_class_counts)}")
    print(f"Class weights saved to: class_weights.pt")
    print(f"Computed mean: {mean_list}")
    print(f"Computed std : {std_list}")
    print("=" * 60 + "\n")

    # visualize one preprocessed image from training set
    try:
        preview_out = cfg.get("preview_out", DEFAULT_CONFIG["preview_out"])
        os.makedirs(os.path.dirname(preview_out) or ".", exist_ok=True)

        # pick a random item from train_dataset
        preview_idx = random.randrange(len(train_dataset))
        img_tensor, lbl = train_dataset[preview_idx]  # already normalized tensor [C,H,W]

        # unnormalize to get values in [0,1] float32
        img_arr = unnormalize_tensor_img(img_tensor, mean, std)

        # img_arr shape can be (H, W) for single-channel or (H, W, C) for multi-channel
        if img_arr.ndim == 3 and img_arr.shape[2] == 1:
            # squeeze the singleton channel axis -> (H, W)
            img_arr = img_arr[:, :, 0]

        # convert to uint8 in [0,255]
        img_uint8 = (img_arr * 255.0).round().astype(np.uint8)

        # create PIL image depending on shape
        if img_uint8.ndim == 2:
            pil_img = Image.fromarray(img_uint8, mode="L")
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
            pil_img = Image.fromarray(img_uint8, mode="RGB")
        else:
            # fallback: attempt to convert problematic shapes to a 3-channel RGB by duplication
            h, w = img_uint8.shape[0], img_uint8.shape[1]

            if img_uint8.ndim == 3 and img_uint8.shape[2] not in (1, 3):
                c = img_uint8.shape[2]

                if c > 3:
                    img_uint8 = img_uint8[:, :, :3]
                else:
                    reps = 3 // c + 1
                    img_uint8 = np.tile(img_uint8, (1, 1, reps))[:, :, :3]
                pil_img = Image.fromarray(img_uint8, mode="RGB")

            else:
                pil_img = Image.fromarray(img_uint8.squeeze(), mode="L")

        # save preview image
        pil_img.save(preview_out)

        print(f"Saved preview image to: {preview_out} (Label: {dataset.idx_to_label[lbl]})")

        # # display
        # try:
        #     import matplotlib.pyplot as plt

        #     plt.figure(figsize=(4, 4))

        #     if img_uint8.ndim == 2:
        #         plt.imshow(img_uint8, cmap='gray', vmin=0, vmax=255)
        #     else:
        #         plt.imshow(img_uint8)

        #     plt.title(f"Preview (Label: {dataset.idx_to_label[lbl]})")
        #     plt.axis("off")
        #     plt.show()
        # except Exception:
        #     print("(Preview display skipped - running in headless environment?)")

    except Exception as e:
        print("Warning: preview generation failed:", e)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": weights,
        "idx_to_label": dataset.idx_to_label,
        "mean": mean,
        "std": std,
        "config": cfg
    }

if __name__ == "__main__":
    out = build_dataloaders()
    print("Done building dataloaders.")