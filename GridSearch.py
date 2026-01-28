import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm

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
		return transforms.Compose([
			transforms.ToTensor(),
			contrast_alpha(alpha=param),
		])
	if transform_type == "noise":
		return transforms.Compose([
			transforms.ToTensor(),
			add_gaussian_noise(sigma=param),
		])
	if transform_type == "blur":
		blur_fn = blur_beta_resample if blur_mode == "resample" else blur_beta_pool
		return transforms.Compose([
			transforms.ToTensor(),
			blur_fn(beta=param),
		])
	raise ValueError(f"Unknown transform type: {transform_type}")


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
) -> None:
	error_list = []
	for n, imgs in tqdm.tqdm(enumerate(dataloader), total=n_iters):
		inputs = imgs
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
) -> torch.Tensor:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
	origin_list = []
	distorted_list = []
	for i, (origin, distorted) in enumerate(dataloader):
		if max_batches is not None and i >= max_batches:
			break
		with torch.no_grad():
			r_origin = infer_olshausen_batch(model, origin, eps, nt_max)
			r_dist = infer_olshausen_batch(model, distorted, eps, nt_max)
		origin_list.append(r_origin.cpu())
		distorted_list.append(r_dist.cpu())
	return torch.cat(origin_list, dim=0), torch.cat(distorted_list, dim=0)


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


def compute_ica_representation(ica: ICA, zca: ZCA, data: np.ndarray) -> np.ndarray:
	whitened = zca.project(data)
	reps = (ica.unprojection_matrix @ whitened.T).T
	return reps.astype(np.float32)


def train_linear_predictor(
	x: torch.Tensor,
	y: torch.Tensor,
	epochs: int,
	lr: float,
	batch_size: int,
	seed: int,
) -> nn.Module:
	torch.manual_seed(seed)
	model = nn.Linear(x.shape[1], y.shape[1])
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()

	loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)
	for _ in range(epochs):
		for xb, yb in loader:
			pred = model(xb)
			loss = loss_fn(pred, yb)
			optim.zero_grad()
			loss.backward()
			optim.step()
	return model


def evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
	loss_fn = nn.MSELoss(reduction="sum")
	loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
	total_loss = 0.0

	with torch.no_grad():
		for xb, yb in loader:
			pred = model(xb)
			total_loss += loss_fn(pred, yb).item()
	return total_loss/len(x)


def split_train_val(x: torch.Tensor, y: torch.Tensor, val_ratio: float, seed: int):
	n = x.shape[0]
	idx = np.arange(n)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)
	split = int(n * (1 - val_ratio))
	train_idx = idx[:split]
	val_idx = idx[split:]
	return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def run_single_transform(
	transform_type: str,
	params: List[float],
	args,
	data: np.ndarray,
	olshausen: OlshausenField1996Model,
	ica: ICA,
	zca: ZCA,
	output_dir: str,
) -> None:
	olshausen_mse = []
	ica_mse = []

	for param in params:
		transform = build_transform(transform_type, param, args.blur_mode)

		indices = np.arange(len(data))
		if args.max_samples is not None:
			indices = indices[:args.max_samples]

		# Olshausen representations
		paired_dataset = PairedPatchesDataset(data, indices, args.imgsize, transform)
		paired_loader = DataLoader(
			paired_dataset,
			batch_size=args.batch_size,
			shuffle=False,
			drop_last=True,
			num_workers=0,
		)
		max_batches = None
		if args.max_samples is not None:
			max_batches = max(1, args.max_samples // args.batch_size)

		r_origin, r_dist = compute_olshausen_reps(
			olshausen, paired_loader, args.eps, args.nt_max, max_batches
		)
		# x_train=r_dist
		# y_train=r_origin
		# x_val = r_dist
		# y_val = r_origin
		x_train, y_train, x_val, y_val = split_train_val(r_dist, r_origin, args.val_ratio, args.seed)
		model_ol = train_linear_predictor(x_train, y_train, args.epochs, args.lr, args.batch_size, args.seed)
		ol_mse = evaluate_mse(model_ol, x_val, y_val, args.batch_size)
		olshausen_mse.append(ol_mse)

		# ICA representations (ZCA whitening)
		data_origin = data[indices]
		data_distorted = apply_transform_numpy(data_origin, args.imgsize, transform)
		r_ica_origin = compute_ica_representation(ica, zca, data_origin)
		zca_distorted = ZCA(input_dim=args.imgsize ** 2)
		zca_distorted.train(data=data_distorted)
		r_ica_dist = compute_ica_representation(ica, zca_distorted, data_distorted)
        
		x_ica = torch.from_numpy(r_ica_dist).float()
		y_ica = torch.from_numpy(r_ica_origin).float()
		x_train, y_train, x_val, y_val = split_train_val(x_ica, y_ica, args.val_ratio, args.seed)
		# x_train=x_ica
		# y_train=y_ica
		# x_val = x_ica
		# y_val = y_ica
		model_ica = train_linear_predictor(x_train, y_train, args.epochs, args.lr, args.batch_size, args.seed)
		ica_mse.append(evaluate_mse(model_ica, x_val, y_val, args.batch_size))

		print(f"{transform_type} param={param}: Olshausen MSE={ol_mse:.6f}, ICA MSE={ica_mse[-1]:.6f}")

	# Save CSV
	csv_path = os.path.join(output_dir, f"{transform_type}_mse.csv")
	with open(csv_path, "w", encoding="utf-8") as f:
		f.write("param,olshausen_mse,ica_mse\n")
		for p, m1, m2 in zip(params, olshausen_mse, ica_mse):
			f.write(f"{p},{m1},{m2}\n")

	# Plot (separate images)
	plt.figure(figsize=(6, 4))
	plt.plot(params, olshausen_mse, marker="o", color="tab:blue")
	plt.xlabel("Transform parameter")
	plt.ylabel("Average MSE")
	plt.title(f"{transform_type} Olshausen prediction MSE")
	plt.tight_layout()
	fig_path = os.path.join(output_dir, f"{transform_type}_olshausen_mse.png")
	plt.savefig(fig_path, dpi=200)
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.plot(params, ica_mse, marker="o", color="tab:orange")
	plt.xlabel("Transform parameter")
	plt.ylabel("Average MSE")
	plt.title(f"{transform_type} ICA prediction MSE")
	plt.tight_layout()
	fig_path = os.path.join(output_dir, f"{transform_type}_ica_mse.png")
	plt.savefig(fig_path, dpi=200)
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Grid search for distortion parameter vs MSE.")
	parser.add_argument("--patches", type=str, default="./patches/patches.npy")
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
	args = parser.parse_args()

	set_seed(args.seed)
	os.makedirs(args.output_dir, exist_ok=True)

	data = load_patches(args.patches, args.imgsize)

	# Train Olshausen model
	olshausen = OlshausenField1996Model(
		num_inputs=args.imgsize ** 2,
		num_units=args.n_units,
		batch_size=args.batch_size,
		device="cpu",
	)
	train_loader = DataLoader(
		torch.from_numpy(data).float(),
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,
	)
	train_olshausen(olshausen, train_loader, args.n_iters, args.eps, args.nt_max)

	# Train ICA model on base data (ZCA whitening)
	zca = ZCA(input_dim=args.imgsize ** 2)
	zca.train(data=data)
	whitened = zca.project(data)
	ica = ICA(input_dim=args.imgsize ** 2)
	ica.train(data=whitened, iterations=args.ica_iterations, convergence=args.ica_convergence, status=False)

	params = parse_param_list(args.params)
	if args.transform == "all":
		for t in ["contrast", "noise", "blur"]:
			run_single_transform(t, params, args, data, olshausen, ica, zca, args.output_dir)
	else:
		run_single_transform(args.transform, params, args, data, olshausen, ica, zca, args.output_dir)


if __name__ == "__main__":
	print("Running GridSearch.py")
	main()
