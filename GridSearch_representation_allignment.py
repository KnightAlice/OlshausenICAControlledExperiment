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
from distortion import contrast_alpha, add_gaussian_noise, blur_beta_pool, blur_beta_resample


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


def build_transform(transform_type: str, param: float, blur_mode: str) -> transforms.Compose:
    if transform_type == "contrast":
        op = contrast_alpha(alpha=param)
    elif transform_type == "noise":
        op = add_gaussian_noise(sigma=param)
    elif transform_type == "blur":
        blur_fn = blur_beta_resample if blur_mode == "resample" else blur_beta_pool
        op = blur_fn(beta=param)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    def _transform(img):
        # Accept ndarray/PIL or torch.Tensor
        if not torch.is_tensor(img):
            img = torch.from_numpy(np.asarray(img)).float()
        return op(img)

    return _transform


class PairedPatchesDataset(Dataset):
	def __init__(self, data: np.ndarray, indices: np.ndarray, imgsize: int, transform) -> None:
		self.data = data
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
		return origin, distorted


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
    for i, (origin, distorted) in enumerate(dataloader):
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
    for origin, distorted in dataloader:
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
) -> nn.Module:
    torch.manual_seed(seed)
    #model = nn.Linear(x.shape[1], y.shape[1]).to(device)
    model = nn.Sequential(
        nn.Softmax(dim=1),
        nn.Linear(x.shape[1], y.shape[1])
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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


def evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> float:
    loss_fn = nn.MSELoss(reduction="sum")
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            #xb = xb.to(device)
            #yb = yb.to(device)
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item()
    return total_loss / len(x)


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
        fontsize=5,
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
    olshausen: OlshausenField1996Model,
    ica: ICA,
    zca: ZCA,
    output_dir: str,
    ssn: Optional[LearnableSSN],
) -> None:
    num_repeats = 5
    olshausen_means = []
    olshausen_stds = []
    ica_means = []
    ica_stds = []
    ssn_means = [] if ssn is not None else None
    ssn_stds = [] if ssn is not None else None

    device = torch.device(args.device)
    
    for param in params:
        flag=0
        transform = build_transform(transform_type, param, args.blur_mode)

        indices = np.arange(len(data))
        if args.max_samples is not None:
            indices = indices[:args.max_samples]

        paired_dataset = PairedPatchesDataset(data, indices, args.imgsize, transform)
        paired_loader = DataLoader(
            paired_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        max_batches = None
        if args.max_samples is not None:
            max_batches = max(1, args.max_samples // args.batch_size)

        r_origin, r_dist = compute_olshausen_reps(
            olshausen, paired_loader, args.eps, args.nt_max, max_batches, device
        )

        r_ssn_origin = None
        r_ssn_dist = None
        if ssn is not None:
            r_ssn_origin, r_ssn_dist = compute_ssn_reps(ssn, paired_loader, device)

        # ICA representations (computed once per param)
        data_origin = data[indices]
        data_distorted = apply_transform_numpy(data_origin, args.imgsize, transform)
        r_ica_origin = compute_ica_representation(ica, None, data_origin, whiten=False)
        r_ica_dist = compute_ica_representation(ica, None, data_distorted, whiten=False)

        x_ica = torch.from_numpy(r_ica_dist).float().to(device)
        y_ica = torch.from_numpy(r_ica_origin).float().to(device)

        ol_runs = []
        ica_runs = []
        ssn_runs = [] if ssn is not None else None

        for rep in range(num_repeats):
            run_seed = args.seed + rep

            x_train, y_train, x_val, y_val = split_train_val(r_dist, r_origin, args.val_ratio, run_seed)
            if flag==0:
                    print("Ol." + str(x_train.max()))
            model_ol = train_linear_predictor(x_train, y_train, args.epochs, args.lr, args.batch_size, run_seed, device)
            ol_runs.append(evaluate_mse(model_ol, x_val, y_val, args.batch_size, device))

            x_train, y_train, x_val, y_val = split_train_val(x_ica, y_ica, args.val_ratio, run_seed)
            if flag==0:
                    print("ICA." + str(x_train.max()))
            model_ica = train_linear_predictor(x_train, y_train, args.epochs, args.lr, args.batch_size, run_seed, device)
            ica_runs.append(evaluate_mse(model_ica, x_val, y_val, args.batch_size, device))

            if ssn is not None:
                x_train, y_train, x_val, y_val = split_train_val(r_ssn_dist, r_ssn_origin, args.val_ratio, run_seed)
                if flag==0:
                    flag=1
                    print("SSN." + str(x_train.max()))

                model_ssn = train_linear_predictor(x_train, y_train, args.epochs, args.lr, args.batch_size, run_seed, device)
                ssn_runs.append(evaluate_mse(model_ssn, x_val, y_val, args.batch_size, device))

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
                f"Olshausen MSE={ol_mean:.6f}±{ol_std:.6f}, "
                f"ICA MSE={ica_mean:.6f}±{ica_std:.6f}"
            )
        else:
            print(
                f"{transform_type} param={param}: "
                f"Olshausen MSE={ol_mean:.6f}±{ol_std:.6f}, "
                f"ICA MSE={ica_mean:.6f}±{ica_std:.6f}, "
                f"SSN MSE={ssn_mean:.6f}±{ssn_std:.6f}"
            )

    # Save CSV
    csv_path = os.path.join(output_dir, f"{transform_type}_mse.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        if ssn is None:
            f.write("param,olshausen_mean,olshausen_std,ica_mean,ica_std\n")
            for p, m1, s1, m2, s2 in zip(params, olshausen_means, olshausen_stds, ica_means, ica_stds):
                f.write(f"{p},{m1},{s1},{m2},{s2}\n")
        else:
            f.write("param,olshausen_mean,olshausen_std,ica_mean,ica_std,ssn_mean,ssn_std\n")
            for p, m1, s1, m2, s2, m3, s3 in zip(params, olshausen_means, olshausen_stds, ica_means, ica_stds, ssn_means, ssn_stds):
                f.write(f"{p},{m1},{s1},{m2},{s2},{m3},{s3}\n")

    # Plot (mean ± std)
    plt.figure(figsize=(8, 5))
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
    ax.set_ylim(bottom=0.0, top=1.0)

    _highlight_original_point(transform_type, params, olshausen_means)
    plt.xlabel("Transform parameter")
    plt.ylabel("Average MSE")
    plt.title(f"{transform_type} prediction MSE (mean ± std, n=5)")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig_path_jpg = os.path.join(output_dir, f"{transform_type}_olshausen_ica_ssn_mse.jpg")
    plt.savefig(fig_path_jpg, dpi=200)
    fig_path_svg = os.path.join(output_dir, f"{transform_type}_olshausen_ica_ssn_mse.svg")
    plt.savefig(fig_path_svg)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for distortion parameter vs MSE.")
    parser.add_argument("--patches", type=str, default="./patches/patches.npy")
    parser.add_argument("--olshausen-patches", type=str, default="./patches/patches_times3.npy")
    parser.add_argument("--imgsize", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-units", type=int, default=100)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--nt-max", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform", type=str, default="all", choices=["contrast", "noise", "blur", "all"])
    parser.add_argument("--params", type=str, default="1,2,3")
    parser.add_argument("--blur-mode", type=str, default="resample", choices=["resample", "pool"])
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--ica-iterations", type=int, default=100)
    parser.add_argument("--ica-convergence", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ssn-checkpoint", type=str, default="./trained_ssn.pth")
    parser.add_argument("--ssn-nE", type=int, default=180)
    parser.add_argument("--ssn-nI", type=int, default=180)
    parser.add_argument("--ssn-T", type=float, default=0.2)
    parser.add_argument("--ssn-dt", type=float, default=1e-3)
    args = parser.parse_args()


    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = torch.device(args.device)

    data = load_patches(args.patches, args.imgsize)
    olshausen_data_path = args.olshausen_patches or args.patches
    data_olshausen = load_patches(olshausen_data_path, args.imgsize)

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
    zca.train(data=data)
    whitened = zca.project(data)
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
    if args.transform == "all":
        for t in ["contrast", "noise", "blur"]:
            run_single_transform(t, params, args, data, olshausen, ica, zca, args.output_dir, ssn)
    else:
        run_single_transform(args.transform, params, args, data, olshausen, ica, zca, args.output_dir, ssn)


if __name__ == "__main__":
    print("Running GridSearch.py")
    main()
