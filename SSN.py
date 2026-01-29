import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio


# -----------------------------
# Utilities: patch sampling
# -----------------------------
def sample_patches(images_np, patch_h, patch_w, batch_size, device="cpu"):
    """
    images_np: (H, W, n_imgs) numpy array
    returns x: (B, D) torch.float32
    """
    H, W, n_imgs = images_np.shape
    ys = np.random.randint(0, H - patch_h + 1, size=batch_size)
    xs = np.random.randint(0, W - patch_w + 1, size=batch_size)
    img_ids = np.random.randint(0, n_imgs, size=batch_size)

    patches = np.empty((batch_size, patch_h, patch_w), dtype=np.float32)
    for i in range(batch_size):
        im = images_np[:, :, img_ids[i]] 
        patches[i] = im[ys[i]:ys[i] + patch_h, xs[i]:xs[i] + patch_w]

    patch_np = patches.reshape(batch_size, -1)
    # patch_np = patch_np - patch_np.mean(axis=1, keepdims=True)  # subtract mean for each patch
    patch_np = patch_np / (patch_np.std() + 1e-8)  # global std normalization

    x = torch.tensor(patch_np, dtype=torch.float32, device=device)

    # random global sign flip (as you specified)
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    x = x * sign
    return x * 3


def gaussian_ring_kernel(n_post, n_pre, sigma_deg=32.0, device="cpu"):
    """
    Build a ring Gaussian kernel based on 'preferred angle index' mapped to [0, 180).
    This keeps your original initialization style, generalized to rectangular matrices.
    """
    # map neuron indices to angles in degrees
    post_angles = torch.linspace(0.0, 180.0, steps=n_post + 1, device=device)[:-1]
    pre_angles  = torch.linspace(0.0, 180.0, steps=n_pre + 1, device=device)[:-1]

    A = post_angles[:, None]  # (n_post,1)
    B = pre_angles[None, :]   # (1,n_pre)
    d = torch.abs(A - B)
    d = torch.minimum(d, 180.0 - d)
    K = torch.exp(-(d * d) / (2.0 * sigma_deg * sigma_deg))
    return K


# -----------------------------
# Plasticity modules
# -----------------------------
class BCMFeedforward:
    """
    BCM: ΔW ~ x * y * (y - theta)
    theta updated by EMA of y^2
    """
    def __init__(self, n_units, d_in, eta=1e-4, theta_lr=1e-3, w_decay=0.0, clamp_nonneg=True):
        self.eta = eta
        self.theta_lr = theta_lr
        self.w_decay = w_decay
        self.clamp_nonneg = clamp_nonneg
        self.theta = torch.zeros(n_units)  # will be moved to device lazily

    def to(self, device):
        self.theta = self.theta.to(device)
        return self

    @torch.no_grad()
    def update(self, W, x_batch, y_bar):
        """
        W: (n_units, d_in)
        x_batch: (B, d_in)
        y_bar: (B, n_units)  time-averaged rates
        """
        device = W.device
        if self.theta.device != device:
            self.to(device)

        # update theta by EMA of y^2 across batch
        y2_mean = (y_bar * y_bar).mean(dim=0)  # (n_units,)
        self.theta = (1.0 - self.theta_lr) * self.theta + self.theta_lr * y2_mean

        # BCM weight update: average over batch
        # ΔW_i = eta * < y_i (y_i - theta_i) x >
        factor = y_bar * (y_bar - self.theta[None, :])  # (B, n_units)
        dW = (factor.T @ x_batch) / x_batch.shape[0]     # (n_units, d_in)

        if self.w_decay > 0:
            dW = dW - self.w_decay * W

        W.add_(self.eta * dW)

        if self.clamp_nonneg:
            W.clamp_(min=0.0)

        # 前向权重归一化（每个神经元的前向权重模长为1）
        with torch.no_grad():
            norms = W.norm(dim=1, keepdim=True) + 1e-8
            W.div_(norms)


class CMRecurrent:
    """
    CM rule (as you wrote):
        Δw = xy - <x><y> (1 + W)
    Here x=pre, y=post. We use batch-mean of yx^T as 'xy',
    and historical EMA means for <x>, <y>.
    If Hz is specified, **add a weight decay (on W) at all [i,j] where both pre/post rates for that sample > Hz**.
    """
    def __init__(self, n_post, n_pre, eta=1e-4, mean_lr=1e-3, clamp_nonneg=True, Hz=None, wd_Hz=1e-1):
        self.eta = eta
        self.mean_lr = mean_lr
        self.clamp_nonneg = clamp_nonneg
        self.Hz = Hz
        self.wd_Hz = wd_Hz
        self.mu_pre  = torch.zeros(n_pre)
        self.mu_post = torch.zeros(n_post)

    def to(self, device):
        self.mu_pre = self.mu_pre.to(device)
        self.mu_post = self.mu_post.to(device)
        return self

    @torch.no_grad()
    def update(self, W, r_pre_bar, r_post_bar, zero_diagonal=False):
        """
        W: (n_post, n_pre)
        r_pre_bar:  (B, n_pre)
        r_post_bar: (B, n_post)
        If self.Hz is not None, apply a weight decay ONLY to W[i,j] where for a given sample,
        both r_post_bar[s, i] > Hz and r_pre_bar[s, j] > Hz (for some s in batch).
        """
        device = W.device
        if self.mu_pre.device != device:
            self.to(device)

        # update running means (EMA) using current batch mean rates
        batch_mu_pre = r_pre_bar.mean(dim=0)
        batch_mu_post = r_post_bar.mean(dim=0)
        self.mu_pre  = (1.0 - self.mean_lr) * self.mu_pre  + self.mean_lr * batch_mu_pre
        self.mu_post = (1.0 - self.mean_lr) * self.mu_post + self.mean_lr * batch_mu_post

        # xy term: < y x^T > over batch
        xy = (r_post_bar.T @ r_pre_bar) / r_pre_bar.shape[0]  # (n_post, n_pre)

        # <y><x> term with (1+W) elementwise
        mu_outer = self.mu_post[:, None] * self.mu_pre[None, :]  # (n_post, n_pre)
        corr = mu_outer * (1.0 + W)

        dW = xy - corr

        # Extra weight decay at [i, j] where for *any* batch sample s, both rates > Hz
        if self.Hz is not None and self.wd_Hz > 0:
            # mask: (B, n_post, n_pre) for "both > Hz"
            post_active = r_post_bar > self.Hz    # (B, n_post)
            pre_active = r_pre_bar > self.Hz      # (B, n_pre)
            joint_mask = post_active[:, :, None] & pre_active[:, None, :]  # (B, n_post, n_pre)
            # For each [i,j], is there any sample where BOTH > Hz?
            mask = joint_mask.any(dim=0)   # (n_post, n_pre) dtype=bool
            # Decay those W[i,j]:
            dW = dW - self.wd_Hz * W * mask

        W.add_(self.eta * dW)

        if zero_diagonal and W.shape[0] == W.shape[1]:
            W.fill_diagonal_(0.0)

        if self.clamp_nonneg:
            W.clamp_(min=0.0)


# -----------------------------
# Learnable SSN (batch dynamics)
# -----------------------------
class LearnableSSN(torch.nn.Module):
    def __init__(
        self,
        nE=180, nI=180,
        patch_h=9, patch_w=9,
        sigma_ori=32.0,
        JEE=0.044, JEI=0.023, JIE=0.042, JII=0.018,
        k=0.04, n=2.0,
        tauE=20e-3, tauI=10e-3,
        dt=1e-3, T=0.2,
        device="cpu",
    ):
        super().__init__()
        self.nE = nE
        self.nI = nI
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.D = patch_h * patch_w
        self.k = k
        self.n = n
        self.tauE = tauE
        self.tauI = tauI
        self.dt = dt
        self.steps = int(T / dt)
        self.device = device

        # ---------- initialize recurrent weights with your ring Gaussian style ----------
        K_EE = gaussian_ring_kernel(nE, nE, sigma_deg=sigma_ori, device=device)
        K_EI = gaussian_ring_kernel(nE, nI, sigma_deg=sigma_ori, device=device)
        K_IE = gaussian_ring_kernel(nI, nE, sigma_deg=sigma_ori, device=device)
        K_II = gaussian_ring_kernel(nI, nI, sigma_deg=sigma_ori, device=device)

        W_EE = JEE * K_EE
        W_EI = JEI * K_EI
        W_IE = JIE * K_IE
        W_II = JII * K_II

        # no self-connection
        W_EE.fill_diagonal_(0.0)
        W_II.fill_diagonal_(0.0)

        # register as parameters (learnable)
        self.W_EE = torch.nn.Parameter(W_EE)
        self.W_EI = torch.nn.Parameter(W_EI)
        self.W_IE = torch.nn.Parameter(W_IE)
        self.W_II = torch.nn.Parameter(W_II)

        # ---------- feedforward receptive fields (learnable) ----------
        # initialize small nonnegative weights and normalize to unit norm
        Wff_E = 0.1 * torch.rand(nE, self.D, device=device)
        Wff_I = 0.1 * torch.rand(nI, self.D, device=device)
        Wff_E = Wff_E / (Wff_E.norm(dim=1, keepdim=True) + 1e-8)
        Wff_I = Wff_I / (Wff_I.norm(dim=1, keepdim=True) + 1e-8)
        self.Wff_E = torch.nn.Parameter(Wff_E)
        self.Wff_I = torch.nn.Parameter(Wff_I)

        # ---------- store initial incoming weight sums for current scaling ----------
        with torch.no_grad():
            self.S_EE0 = 3.75 #self.W_EE.sum(dim=1).clone()  # (nE,)
            self.S_EI0 = 1.86 #self.W_EI.sum(dim=1).clone()  # (nE,)
            self.S_IE0 = 3.13  #self.W_IE.sum(dim=1).clone()  # (nI,)
            self.S_II0 = 1.11 - 0.02 #self.W_II.sum(dim=1).clone()  # (nI,)

        self.eps = 1e-12

        # ---------- pluggable plasticity hooks ----------
        # default: BCM for FF, CM for all recurrent
        self.plastic_ff_E = BCMFeedforward(n_units=nE, d_in=self.D, eta=1e-4, theta_lr=1e-3,clamp_nonneg=False)
        self.plastic_ff_I = BCMFeedforward(n_units=nI, d_in=self.D, eta=1e-4, theta_lr=1e-3,clamp_nonneg=False)

        self.plastic_EE = CMRecurrent(n_post=nE, n_pre=nE, eta=1e-4, mean_lr=1e-3)
        self.plastic_EI = CMRecurrent(n_post=nE, n_pre=nI, eta=1e-4, mean_lr=1e-3)
        self.plastic_IE = CMRecurrent(n_post=nI, n_pre=nE, eta=1e-4, mean_lr=1e-3)
        self.plastic_II = CMRecurrent(n_post=nI, n_pre=nI, eta=1e-4, mean_lr=1e-3)

    def powerlaw_io(self, I):
        return self.k * torch.clamp(I, min=0.0) ** self.n

    @torch.no_grad()
    def current_scalers(self):
        """
        Return per-postsyn scaling factors for recurrent E and I currents.
        """
        S_EE = self.W_EE.sum(dim=1)
        S_EI = self.W_EI.sum(dim=1)
        S_IE = self.W_IE.sum(dim=1)
        S_II = self.W_II.sum(dim=1)

        gE_E = self.S_EE0 / (S_EE + self.eps)  # (nE,)
        gI_E = self.S_EI0 / (S_EI + self.eps)  # (nE,)
        gE_I = self.S_IE0 / (S_IE + self.eps)  # (nI,)
        gI_I = self.S_II0 / (S_II + self.eps)  # (nI,)
        return gE_E, gI_E, gE_I, gI_I

    def forward_dynamics(self, x_batch, record=False):
        """
        x_batch: (B, D) natural image patches
        Returns time-averaged rates and (optionally) time-averaged currents for monitoring.
        """
        B = x_batch.shape[0]
        device = x_batch.device

        # feedforward currents
        Iff_E = x_batch @ self.Wff_E.T  # (B, nE)
        Iff_I = x_batch @ self.Wff_I.T  # (B, nI)

        # initialize rates
        rE = torch.zeros(B, self.nE, device=device)
        rI = torch.zeros(B, self.nI, device=device)

        # for time averages
        rE_acc = torch.zeros_like(rE)
        rI_acc = torch.zeros_like(rI)

        # monitoring currents (time-avg)
        if record:
            IffE_acc = torch.zeros_like(rE)
            IffI_acc = torch.zeros_like(rI)
            IrecE_exc_acc = torch.zeros_like(rE)
            IrecE_inh_acc = torch.zeros_like(rE)
            IrecI_exc_acc = torch.zeros_like(rI)
            IrecI_inh_acc = torch.zeros_like(rI)

        gE_E, gI_E, gE_I, gI_I = self.current_scalers()  # (nE,), (nE,), (nI,), (nI,)
        self.W_EE.data.clamp_(max=0.15)
        self.W_EI.data.clamp_(min=0.0005)
        # 直接缩放权重
        self.W_EE.data = self.W_EE.data * gE_E[:, None]
        self.W_EI.data = self.W_EI.data * gI_E[:, None]
        self.W_IE.data = self.W_IE.data * gE_I[:, None]
        self.W_II.data = self.W_II.data * gI_I[:, None]

        # 记录最后50步的放电率求平均作为bar
        last_N = 50
        rE_buffer = []
        rI_buffer = []

        for step in range(self.steps):
            # recurrent currents, now using缩放后的权重
            IrecE_exc = rE @ self.W_EE.T              # (B, nE)
            IrecE_inh = rI @ self.W_EI.T              # (B, nE)
            IrecI_exc = rE @ self.W_IE.T              # (B, nI)
            IrecI_inh = rI @ self.W_II.T              # (B, nI)

            # total input currents
            IE = Iff_E + IrecE_exc - IrecE_inh
            II = Iff_I + IrecI_exc - IrecI_inh

            rE_ss = self.powerlaw_io(IE)
            rI_ss = self.powerlaw_io(II)

            rE = rE + (self.dt / self.tauE) * (-rE + rE_ss)
            rI = rI + (self.dt / self.tauI) * (-rI + rI_ss)

            # Maintain buffer of the last N firing rates
            if self.steps - step <= last_N:
                rE_buffer.append(rE.clone())
                rI_buffer.append(rI.clone())

            if record:
                IffE_acc += Iff_E
                IffI_acc += Iff_I
                IrecE_exc_acc += IrecE_exc
                IrecE_inh_acc += IrecE_inh
                IrecI_exc_acc += IrecI_exc
                IrecI_inh_acc += IrecI_inh

        if len(rE_buffer) > 0 and len(rI_buffer) > 0:
            rE_bar = torch.stack(rE_buffer, dim=0).mean(dim=0)
            rI_bar = torch.stack(rI_buffer, dim=0).mean(dim=0)
        else:
            rE_bar = rE
            rI_bar = rI

        if not record:
            return rE_bar, rI_bar, None

        monitor = {
            "Iff_E_bar": IffE_acc / self.steps,
            "Iff_I_bar": IffI_acc / self.steps,
            "IrecE_exc_bar": IrecE_exc_acc / self.steps,
            "IrecE_inh_bar": IrecE_inh_acc / self.steps,
            "IrecI_exc_bar": IrecI_exc_acc / self.steps,
            "IrecI_inh_bar": IrecI_inh_acc / self.steps,
        }
        return rE_bar, rI_bar, monitor

    @torch.no_grad()
    def plasticity_step(self, x_batch, rE_bar, rI_bar):
        """
        Apply plasticity using time-averaged rates per sample.
        """
        # feedforward BCM
        self.plastic_ff_E.update(self.Wff_E, x_batch, rE_bar)
        self.plastic_ff_I.update(self.Wff_I, x_batch, rI_bar)

        # 归一化前向权重（每神经元模长为1）
        self.Wff_E.div_(self.Wff_E.norm(dim=1, keepdim=True) + 1e-8)
        self.Wff_I.div_(self.Wff_I.norm(dim=1, keepdim=True) + 1e-8)

        # recurrent CM (pluggable)
        self.plastic_EE.update(self.W_EE, rE_bar, rE_bar, zero_diagonal=True)
        self.plastic_EI.update(self.W_EI, rI_bar, rE_bar, zero_diagonal=False)
        self.plastic_IE.update(self.W_IE, rE_bar, rI_bar, zero_diagonal=False)
        self.plastic_II.update(self.W_II, rI_bar, rI_bar, zero_diagonal=True)

    @torch.no_grad()
    def summary_stats(self, rE_bar, rI_bar, monitor):
        """
        Real-time monitoring you requested: mean/max firing rates, mean currents, etc.
        """
        out = {}
        out["rE_mean"] = rE_bar.mean().item()
        out["rE_max"]  = rE_bar.max().item()
        out["rI_mean"] = rI_bar.mean().item()
        out["rI_max"]  = rI_bar.max().item()

        if monitor is not None:
            out["Iff_E_mean"] = monitor["Iff_E_bar"].mean().item()
            out["Iff_I_mean"] = monitor["Iff_I_bar"].mean().item()
            out["IrecE_exc_mean"] = monitor["IrecE_exc_bar"].mean().item()
            out["IrecE_inh_mean"] = monitor["IrecE_inh_bar"].mean().item()
            out["IrecI_exc_mean"] = monitor["IrecI_exc_bar"].mean().item()
            out["IrecI_inh_mean"] = monitor["IrecI_inh_bar"].mean().item()

        # also monitor incoming weight sums (to verify scaling mechanism stays finite)
        out["sumW_EE_mean"] = self.W_EE.sum(dim=1).mean().item()
        out["sumW_EI_mean"] = self.W_EI.sum(dim=1).mean().item()
        out["sumW_IE_mean"] = self.W_IE.sum(dim=1).mean().item()
        out["sumW_II_mean"] = self.W_II.sum(dim=1).mean().item()

        return out
    
    def get_final_state(self, x_batch):
        """
        x_batch: (B, D) natural image patches
        只返回在网络动力学最后时刻（最后一步 step）的所有状态变量（以及所有电流）。
        """
        B = x_batch.shape[0]
        device = x_batch.device

        # feedforward currents
        Iff_E = x_batch @ self.Wff_E.T  # (B, nE)
        Iff_I = x_batch @ self.Wff_I.T  # (B, nI)

        # initialize rates
        rE = torch.zeros(B, self.nE, device=device)
        rI = torch.zeros(B, self.nI, device=device)

        # gE_E, gI_E, gE_I, gI_I = self.current_scalers()
        # W_EE = self.W_EE.clamp(max=0.15) * gE_E[:, None]
        # W_EI = self.W_EI.clamp(min=0.0005) * gI_E[:, None]
        # W_IE = self.W_IE * gE_I[:, None]
        # W_II = self.W_II * gI_I[:, None]

        for step in range(self.steps):
            # recurrent currents (用缩放后的权重)
            IrecE_exc = rE @ self.W_EE.T              # (B, nE)
            IrecE_inh = rI @self.W_EI.T              # (B, nE)
            IrecI_exc = rE @ self.W_IE.T              # (B, nI)
            IrecI_inh = rI @ self.W_II.T              # (B, nI)

            # total input currents
            IE = Iff_E + IrecE_exc - IrecE_inh
            II = Iff_I + IrecI_exc - IrecI_inh

            rE_ss = self.powerlaw_io(IE)
            rI_ss = self.powerlaw_io(II)

            rE = rE + (self.dt / self.tauE) * (-rE + rE_ss)
            rI = rI + (self.dt / self.tauI) * (-rI + rI_ss)
            
        return rE, rI,None



# -----------------------------
# Training loop (minimal)
# -----------------------------
def train_ssn(
    MAT_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu",
    patch_h=9, patch_w=9,
    batch_size=64,
    epochs=50,
    steps_per_epoch=100,
):
    mat = sio.loadmat(MAT_PATH)
    images = mat["IMAGES"]  # (512,512,10) typically
    images = np.array(images, dtype=np.float32)

    ssn = LearnableSSN(
        nE=180, nI=180,
        patch_h=patch_h, patch_w=patch_w,
        T=0.2, dt=1e-3,
        device=device
    ).to(device)

    for ep in range(1, epochs + 1):
        for it in range(1, steps_per_epoch + 1):
            x = sample_patches(images, patch_h, patch_w, batch_size, device=device)

            rE_bar, rI_bar, monitor = ssn.forward_dynamics(x, record=True)
            ssn.plasticity_step(x, rE_bar, rI_bar)
            # 前向权重归一化，确保每个神经元的Wff模长等于1
            with torch.no_grad():
                ssn.Wff_E.div_(ssn.Wff_E.norm(dim=1, keepdim=True) + 1e-8)
                ssn.Wff_I.div_(ssn.Wff_I.norm(dim=1, keepdim=True) + 1e-8)
            stats = ssn.summary_stats(rE_bar, rI_bar, monitor)
            if it % steps_per_epoch == 0:
                print(
                    f"[Ep {ep:03d} It {it:04d}] "
                    f"rE mean/max={stats['rE_mean']:.4f}/{stats['rE_max']:.4f} | "
                    f"rI mean/max={stats['rI_mean']:.4f}/{stats['rI_max']:.4f} | "
                    f"IffE/IrecEexc/IrecEinh mean="
                    f"{stats['Iff_E_mean']:.4f}/{stats['IrecE_exc_mean']:.4f}/{stats['IrecE_inh_mean']:.4f} | "
                    f"sumW_EE/EI/IE/II mean="
                    f"{stats['sumW_EE_mean']:.4f}/{stats['sumW_EI_mean']:.4f}/"
                    f"{stats['sumW_IE_mean']:.4f}/{stats['sumW_II_mean']:.4f}"
                )

    return ssn

if __name__ == "__main__":
    MAT_PATH = r".\datasets\IMAGES.mat"
    ssn = train_ssn(MAT_PATH, batch_size=1024, epochs=50, steps_per_epoch=20)