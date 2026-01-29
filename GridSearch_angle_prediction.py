import argparse
import json
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm
from SSN import LearnableSSN

from network import OlshausenField1996Model
from pydeep.preprocessing import ICA, ZCA
from distortion import contrast_alpha, add_gaussian_noise, blur_beta_pool, blur_beta_resample, GaussianBlur


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def parse_param_list(param_str: str) -> List[float]:
	"""
	Support "a,b,c" or "start:stop:step".
	"""
	if ":" in param_str:
		start, stop, step = param_str.split(":")
		start = float(start)
		stop = float(stop)
		step = float(step)
		return list(np.arange(start, stop + 1e-9, step))
	return [float(x) for x in param_str.split(",") if x.strip() != ""]


def load_patches(path: str, imgsize: int) -> np.ndarray:
	data = np.load(path).squeeze().reshape(-1, imgsize * imgsize)
	return data.astype(np.float32)


def gabor_like(size: int, theta: float, sigma: float, freq: float, phase: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2+ 1)
    xx, yy = np.meshgrid(ax, ax)
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    gauss = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
    sinusoid = np.cos(2 * np.pi * freq * x_theta + phase)
    gabor = gauss * sinusoid
    return gabor.astype(np.float32)


def generate_gabor_dataset(
    imgsize: int,
    n_samples: int,
    angle_jitter_deg: float,
    sigma: float,
    freq: float,
    phase: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    angles = {
        "vertical": np.deg2rad(90),
        "45": np.deg2rad(45),
        "horizontal": np.deg2rad(0),
        "135": np.deg2rad(135),
    }
    labels = {"horizontal": 0, "45": 1, "vertical": 2, "135": 3}
    angle_names = list(labels.keys())

    rng = np.random.default_rng(seed)
    X = []
    y = []
    for _ in range(n_samples):
        name = rng.choice(angle_names)
        base_angle = angles[name]
        delta = np.deg2rad(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
        kernel = gabor_like(size=imgsize, theta=base_angle + delta, sigma=sigma, freq=freq, phase=phase)
        X.append(kernel.reshape(-1))
        y.append(labels[name])

    X = np.stack(X, axis=0).astype(np.float32)*20
    y = np.array(y, dtype=np.int64)
    # print(X.max(), X.min(), X.mean())
    # raise
    return X, y, len(labels), angle_names


def build_transform(transform_type: str, param: float, blur_mode: str) -> transforms.Compose:
    if transform_type == "contrast":
        op = contrast_alpha(alpha=param)
    elif transform_type == "noise":
        op = add_gaussian_noise(sigma=param)
    elif transform_type == "blur":
        if blur_mode == "resample":
            blur_fn = blur_beta_resample
        elif blur_mode == "pool":
            blur_fn = blur_beta_pool
        elif blur_mode == "gaussian":
            blur_fn = GaussianBlur
        op = blur_fn(param)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    def _transform(img):
        # Accept ndarray/PIL or torch.Tensor
        if not torch.is_tensor(img):
            img = torch.from_numpy(np.asarray(img)).float()
        return op(img)

    return _transform


class PairedPatchesDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: np.ndarray, imgsize: int, transform) -> None:
        self.data = data
        self.labels = labels
        self.indices = indices
        self.imgsize = imgsize
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        patch = self.data[self.indices[idx]]
        img = patch.reshape(self.imgsize, self.imgsize)
        origin = torch.from_numpy(patch).float()
        distorted = self.transform(img).view(-1).float()
        label = torch.tensor(self.labels[self.indices[idx]], dtype=torch.long)
        return origin, distorted, label


def train_olshausen(
    model: OlshausenField1996Model,
    dataloader: DataLoader,
    n_iters: int,
    eps: float,
    nt_max: int,
    device: torch.device,
) -> None:
    error_list = []
    for n, imgs in tqdm.tqdm(enumerate(dataloader), total=n_iters):
        inputs = imgs.to(device)
        if inputs.shape[1] != model.num_inputs:
            continue
        model.initialize_states()
        model.normalize_rows()
        r_tm1 = model.r

        for t in range(nt_max):
            error, r = model(inputs, training=False)
            dr = r - r_tm1
            dr_norm = torch.norm(dr) / (eps + torch.norm(r_tm1))
            r_tm1 = r
            if dr_norm < eps:
                error, r = model(inputs, training=True)
                break
            if t >= nt_max - 2:
                break

        error_list.append(model.calculate_total_error(error))
        if n >= n_iters:
            break


def infer_olshausen_batch(
    model: OlshausenField1996Model,
    inputs: torch.Tensor,
    eps: float,
    nt_max: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = inputs.to(device)
    model.initialize_states()
    model.normalize_rows()
    r_tm1 = model.r

    for _ in range(nt_max):
        _, r = model(inputs, training=False)
        dr = r - r_tm1
        dr_norm = torch.norm(dr) / (eps + torch.norm(r_tm1))
        r_tm1 = r
        if dr_norm < eps:
            break

    return r


def compute_olshausen_reps(
    model: OlshausenField1996Model,
    dataloader: DataLoader,
    eps: float,
    nt_max: int,
    max_batches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    origin_list = []
    distorted_list = []
    for i, (origin, distorted, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.no_grad():
            r_origin = infer_olshausen_batch(model, origin, eps, nt_max, device)
            r_dist = infer_olshausen_batch(model, distorted, eps, nt_max, device)
        origin_list.append(r_origin.cpu())
        distorted_list.append(r_dist.cpu())
    return torch.cat(origin_list, dim=0).to(device), torch.cat(distorted_list, dim=0).to(device)


def compute_ssn_reps(
    ssn: LearnableSSN,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    origin_list = []
    distorted_list = []
    for origin, distorted, _ in dataloader:
        origin = origin.to(device)
        distorted = distorted.to(device)
        with torch.no_grad():
            r_origin, _, _ = ssn.get_final_state(origin)
            r_dist, _, _ = ssn.get_final_state(distorted)
        origin_list.append(r_origin)
        distorted_list.append(r_dist)
    return torch.cat(origin_list, dim=0).to(device), torch.cat(distorted_list, dim=0).to(device)


def apply_transform_numpy(
	data: np.ndarray,
	imgsize: int,
	transform,
) -> np.ndarray:
	distorted = []
	for patch in data:
		img = patch.reshape(imgsize, imgsize)
		d = transform(img).view(-1).numpy().astype(np.float32)
		distorted.append(d)
	return np.stack(distorted, axis=0)


def compute_ica_representation(ica: ICA, zca: ZCA, data: np.ndarray, whiten: bool = True) -> np.ndarray:
	if whiten:
		whitened = zca.project(data)
	else: #when it comes to compare representations, use original data to keep data space consistent
		whitened = data
	reps = (ica.unprojection_matrix @ whitened.T).T
	return reps.astype(np.float32)


def train_linear_predictor(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_classes: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], num_classes).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for xb, yb in loader:
            #xb = xb.to(device)
            #yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            #xb = xb.to(device)
            #yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	idx = np.arange(n)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)
	split = int(n * (1 - val_ratio))
	return idx[:split], idx[split:]


def split_train_val(x: torch.Tensor, y: torch.Tensor, val_ratio: float, seed: int):
	n = x.shape[0]
	idx = np.arange(n)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)
	split = int(n * (1 - val_ratio))
	train_idx = idx[:split]
	val_idx = idx[split:]
	return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _highlight_original_point(transform_type: str, params: List[float], values: List[float]) -> None:
    target = None
    if transform_type == "contrast":
        target = 1.0
    elif transform_type == "blur":
        target = 0.0
    elif transform_type == "noise":
        target = 0.0
    else:
        return

    if target is None:
        return

    params_arr = np.asarray(params, dtype=np.float32)
    idx = np.where(np.isclose(params_arr, target, atol=1e-6))[0]
    if idx.size == 0:
        return

    i = int(idx[0])
    plt.scatter([params[i]], [values[i]], color="tab:red", s=60, zorder=5)
    plt.text(
        params[i],
        values[i],
        "Orig.",
        fontsize=7,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontstyle="italic",
        zorder=10,
    )

def run_single_transform(
    transform_type: str,
    params: List[float],
    args,
    data: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    olshausen: OlshausenField1996Model,
    ica: ICA,
    zca: ZCA,
    output_dir: str,
    ssn: Optional[LearnableSSN],
) -> None:
    num_repeats = args.num_repeats
    olshausen_means = []
    olshausen_stds = []
    ica_means = []
    ica_stds = []
    ssn_means = [] if ssn is not None else None
    ssn_stds = [] if ssn is not None else None

    device = torch.device(args.device)
    all_transforms = ["contrast", "noise", "blur"]

    # Data preparation (training)
    rng = np.random.default_rng(args.seed)
    train_ol_list = []
    train_ica_list = []
    train_label_list = []
    train_ssn_list = [] if ssn is not None else None

    for t in all_transforms:
        train_params = parse_param_list(getattr(args, f"train_params_{t}"))
        for training_param in train_params:
            sample_count = max(args.samples_per_param, args.batch_size)
            indices = rng.choice(len(data), size=sample_count, replace=sample_count > len(data))
            usable = (len(indices) // args.batch_size) * args.batch_size
            indices = indices[:usable]
            if len(indices) == 0:
                continue

            t_transform = build_transform(t, training_param, args.blur_mode)
            t_dataset = PairedPatchesDataset(data, labels, indices, args.imgsize, t_transform)
            t_loader = DataLoader(
                t_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            _, r_dist_t = compute_olshausen_reps(
                olshausen, t_loader, args.eps, args.nt_max, None, device
            )
            train_ol_list.append(r_dist_t)

            if ssn is not None:
                _, r_ssn_dist_t = compute_ssn_reps(ssn, t_loader, device)
                train_ssn_list.append(r_ssn_dist_t)

            data_origin = data[indices]
            data_distorted = apply_transform_numpy(data_origin, args.imgsize, t_transform)
            r_ica_dist_t = compute_ica_representation(ica, None, data_distorted, whiten=False)
            train_ica_list.append(torch.from_numpy(r_ica_dist_t).float().to(device))
            train_label_list.append(torch.from_numpy(labels[indices]).long().to(device))

    if len(train_label_list) == 0:
        return

    x_train_ol = torch.cat(train_ol_list, dim=0)
    x_train_ica = torch.cat(train_ica_list, dim=0)
    y_train_all = torch.cat(train_label_list, dim=0)
    x_train_ssn = torch.cat(train_ssn_list, dim=0) if ssn is not None else None

    
    
    for param in params:
        sample_count = max(args.max_samples, args.batch_size)
        eval_indices = rng.choice(len(data), size=sample_count, replace=sample_count > len(data))
        usable = (len(eval_indices) // args.batch_size) * args.batch_size
        eval_indices = eval_indices[:usable]
        if len(eval_indices) == 0:
            continue

        label_subset = torch.from_numpy(labels[eval_indices]).long().to(device)

        eval_transform = build_transform(transform_type, param, args.blur_mode)
        eval_dataset = PairedPatchesDataset(data, labels, eval_indices, args.imgsize, eval_transform)
 
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        _, r_dist = compute_olshausen_reps(
            olshausen, eval_loader, args.eps, args.nt_max, None, device
        )

        r_ssn_dist = None
        if ssn is not None:
            _, r_ssn_dist = compute_ssn_reps(ssn, eval_loader, device)

        data_origin = data[eval_indices]
        data_distorted = apply_transform_numpy(data_origin, args.imgsize, eval_transform)
        r_ica_dist = compute_ica_representation(ica, None, data_distorted, whiten=False)
        x_ica = torch.from_numpy(r_ica_dist).float().to(device)

        ol_runs = []
        ica_runs = []
        ssn_runs = [] if ssn is not None else None

        for rep in range(num_repeats):
            run_seed = args.seed + rep

            model_ol = train_linear_predictor(
                        x_train_ol, y_train_all, args.epochs, args.lr, args.batch_size, run_seed, device, num_classes
                    )
            
            model_ica = train_linear_predictor(
                        x_train_ica, y_train_all, args.epochs, args.lr, args.batch_size, run_seed, device, num_classes
                    )
            
            model_ssn = train_linear_predictor(
                            x_train_ssn, y_train_all, args.epochs, args.lr, args.batch_size, run_seed, device, num_classes
                        )
            
            
            ol_runs.append(evaluate_accuracy(model_ol, r_dist, label_subset, args.batch_size, device))

            
            ica_runs.append(evaluate_accuracy(model_ica, x_ica, label_subset, args.batch_size, device))

            if ssn is not None:
                
                ssn_runs.append(evaluate_accuracy(model_ssn, r_ssn_dist, label_subset, args.batch_size, device))

        ol_mean = float(np.mean(ol_runs))
        ol_std = float(np.std(ol_runs))
        ica_mean = float(np.mean(ica_runs))
        ica_std = float(np.std(ica_runs))

        olshausen_means.append(ol_mean)
        olshausen_stds.append(ol_std)
        ica_means.append(ica_mean)
        ica_stds.append(ica_std)

        if ssn is not None:
            ssn_mean = float(np.mean(ssn_runs))
            ssn_std = float(np.std(ssn_runs))
            ssn_means.append(ssn_mean)
            ssn_stds.append(ssn_std)

        if ssn is None:
            print(
                f"{transform_type} param={param}: "
                f"Olshausen Acc={ol_mean:.6f}±{ol_std:.6f}, "
                f"ICA Acc={ica_mean:.6f}±{ica_std:.6f}"
            )
        else:
            print(
                f"{transform_type} param={param}: "
                f"Olshausen Acc={ol_mean:.6f}±{ol_std:.6f}, "
                f"ICA Acc={ica_mean:.6f}±{ica_std:.6f}, "
                f"SSN Acc={ssn_mean:.6f}±{ssn_std:.6f}"
            )

    # Save CSV
    csv_path = os.path.join(output_dir, f"{transform_type}_acc.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        if ssn is None:
            f.write("param,olshausen_acc_mean,olshausen_acc_std,ica_acc_mean,ica_acc_std\n")
            for p, m1, s1, m2, s2 in zip(params, olshausen_means, olshausen_stds, ica_means, ica_stds):
                f.write(f"{p},{m1},{s1},{m2},{s2}\n")
        else:
            f.write("param,olshausen_acc_mean,olshausen_acc_std,ica_acc_mean,ica_acc_std,ssn_acc_mean,ssn_acc_std\n")
            for p, m1, s1, m2, s2, m3, s3 in zip(params, olshausen_means, olshausen_stds, ica_means, ica_stds, ssn_means, ssn_stds):
                f.write(f"{p},{m1},{s1},{m2},{s2},{m3},{s3}\n")

    # Plot (mean ± std)
    plt.figure(figsize=(3, 3))
    plt.axhline(y=1.0 / num_classes, color="gray", linestyle="--", label="Chance Level")
    plt.plot(params, olshausen_means, marker="o", color="tab:blue", markersize=3, label="Olshausen")
    plt.fill_between(
        params,
        np.asarray(olshausen_means) - np.asarray(olshausen_stds),
        np.asarray(olshausen_means) + np.asarray(olshausen_stds),
        color="tab:blue",
        alpha=0.2,
    )
    plt.plot(params, ica_means, marker="o", color="tab:orange", markersize=3, label="ICA")
    plt.fill_between(
        params,
        np.asarray(ica_means) - np.asarray(ica_stds),
        np.asarray(ica_means) + np.asarray(ica_stds),
        color="tab:orange",
        alpha=0.2,
    )
    if ssn is not None:
        plt.plot(params, ssn_means, marker="o", color="tab:green", markersize=3, label="SSN")
        plt.fill_between(
            params,
            np.asarray(ssn_means) - np.asarray(ssn_stds),
            np.asarray(ssn_means) + np.asarray(ssn_stds),
            color="tab:green",
            alpha=0.2,
        )
    ax = plt.gca()
    #ax.set_ylim(bottom=0.0, top=1.0)
    if transform_type=='noise':
         ax.set_ylim(bottom=0.2)
    ax.set_ylim(bottom=0.0)

    _highlight_original_point(transform_type, params, olshausen_means)
    if transform_type == "noise":
        plt.xlabel("Noise factor (σ)")
    elif transform_type == "contrast":
        plt.xlabel("Contrast factor (α)")
    elif transform_type == "blur":
        plt.xlabel("Blur factor (β)")
    plt.ylabel("Average Accuracy")
    plt.title(f"Distortion type:{transform_type}, Acc. (mean ± std, n={args.num_repeats})")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig_path_jpg = os.path.join(output_dir, f"{transform_type}_olshausen_ica_ssn_acc.jpg")
    plt.savefig(fig_path_jpg, dpi=200)
    fig_path_svg = os.path.join(output_dir, f"{transform_type}_olshausen_ica_ssn_acc.svg")
    plt.savefig(fig_path_svg)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for distortion parameter vs accuracy.")
    parser.add_argument("--patches", type=str, default="./patches/patches.npy")
    parser.add_argument("--olshausen-patches", type=str, default="./patches/patches_times3.npy")
    parser.add_argument("--imgsize", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-units", type=int, default=100)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--num-repeats", type=int, default=5,help="Number of repeats for each parameter setting.")
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--nt-max", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum number of samples to use during evaluation.")
    parser.add_argument("--samples-per-param", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform", type=str, default="all", choices=["contrast", "noise", "blur", "all"])
    parser.add_argument("--params", type=str, default="1,2,3")
    parser.add_argument("--train-params-contrast", type=str, default="0,0.001,0.002,0.003,0.004,0.005,0.0075,0.01",help="Parameters used during training for contrast.")
    parser.add_argument("--train-params-noise", type=str, default="0,1,1.5",help="Parameters used during training for noise.")
    parser.add_argument("--train-params-blur", type=str, default="0.5,1,1.5,",help="Parameters used during training for blur./Original")
    #parser.add_argument("--train-params-blur", type=str, default="0,1,2,2,5,3,3.5,4,4.5",help="Parameters used during training for blur.")
    parser.add_argument("--blur-mode", type=str, default="resample", choices=["resample", "pool","gaussian"], help="Mode used for blurring.")
    parser.add_argument("--output-dir", type=str, default="./outputs_angle")
    parser.add_argument("--ica-iterations", type=int, default=100)
    parser.add_argument("--ica-convergence", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ssn-checkpoint", type=str, default="./trained_ssn.pth")
    parser.add_argument("--ssn-nE", type=int, default=180)
    parser.add_argument("--ssn-nI", type=int, default=180)
    parser.add_argument("--ssn-T", type=float, default=0.2)
    parser.add_argument("--ssn-dt", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--angle-jitter-deg", type=float, default=10.0)
    parser.add_argument("--gabor-sigma", type=float, default=2.0)
    parser.add_argument("--gabor-freq", type=float, default=0.3)
    parser.add_argument("--gabor-phase", type=float, default=0.0)
    args = parser.parse_args()


    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = torch.device(args.device)

    data, labels, num_classes, _ = generate_gabor_dataset(
        imgsize=args.imgsize,
        n_samples=args.n_samples,
        angle_jitter_deg=args.angle_jitter_deg,
        sigma=args.gabor_sigma,
        freq=args.gabor_freq,
        phase=args.gabor_phase,
        seed=args.seed,
    )
    olshausen_data_path = args.olshausen_patches
    data_olshausen = load_patches(olshausen_data_path, args.imgsize)
    data_ica = load_patches(args.patches, args.imgsize)

    # Train Olshausen model
    olshausen = OlshausenField1996Model(
        num_inputs=args.imgsize ** 2,
        num_units=args.n_units,
        batch_size=args.batch_size,
        device=args.device,
    )
    train_loader = DataLoader(
        torch.from_numpy(data_olshausen).float(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    train_olshausen(olshausen, train_loader, args.n_iters, args.eps, args.nt_max, device)

    # Train ICA model on base data (ZCA whitening)
    zca = ZCA(input_dim=args.imgsize ** 2)
    zca.train(data=data_ica)
    whitened = zca.project(data_ica)
    ica = ICA(input_dim=args.imgsize ** 2)
    ica.train(data=whitened, iterations=args.ica_iterations, convergence=args.ica_convergence, status=False)

    # Load SSN (optional)
    ssn = None
    if args.ssn_checkpoint and os.path.exists(args.ssn_checkpoint):
        ssn = LearnableSSN(
            nE=args.ssn_nE,
            nI=args.ssn_nI,
            patch_h=args.imgsize,
            patch_w=args.imgsize,
            T=args.ssn_T,
            dt=args.ssn_dt,
            device=args.device,
        ).to(device)
        ssn.load_state_dict(torch.load(args.ssn_checkpoint, map_location=device))
        ssn.training = False
    else:
        print("SSN checkpoint not found, skipping SSN comparison.")

    params = parse_param_list(args.params)
    run_single_transform(args.transform, params, args, data, labels, num_classes, olshausen, ica, zca, args.output_dir, ssn)


if __name__ == "__main__":
    print("Running GridSearch.py")
    main()
