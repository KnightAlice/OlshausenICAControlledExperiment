import torch
import torch.nn.functional as F


import warnings

class Normalize01():
    def __call__(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        return normalized_tensor

class contrast_alpha():
    """
    I, alpha, alpha_max=10.0, center='mean'
    """
    def __init__(self, alpha=1.0, alpha_max=10.0, center='mean'):
        self.alpha = alpha
        self.alpha_max = alpha_max
        self.center = center

    def __call__(self, I):
        """
        alpha>=0.0 and alpha<=alpha_max
        """
        # choose center
        if self.center == 'mean':
            mu = I.mean()
        elif self.center == 'median':
            mu = I.median()
        else:
            raise ValueError

        # linear contrast scaling (NO CLAMP)
        base = mu + self.alpha * (I - mu)

        if self.alpha <= 1:
            return base

        # explicit binary endpoint
        B = (I > mu).float()
        lam = (self.alpha - 1) / (self.alpha_max - 1)
        lam = torch.clamp(torch.tensor(lam), 0., 1.)

        return (1 - lam) * base + lam * B

class add_gaussian_noise:
    """
    sigma: in [0,inf]
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, I):
        sigma_I = I.std()
        noise = self.sigma * torch.randn_like(I) * sigma_I
        return I + noise


class blur_beta_pool:
    """
    Avgpool blur.
    I,
    beta>=0.0
    """
    def __init__(self,beta=0):
        self.beta = beta

    def __call__(self, I):

        if self.beta == 0:
            return I
        
        if I.dim() == 2:
            H, W = I.shape
            I = I.unsqueeze(0)
        elif I.dim() == 3:
            _, H, W = I.shape

        k = int(1 + self.beta * min(H, W) / 20)
        if k % 2 == 0:
            k += 1

        return F.avg_pool2d(I, kernel_size=(k,k), stride=1, padding=k//2).squeeze(0)
    

class blur_beta_resample:
    """
    Edge-blur via downsample + upsample(bicubic).
    beta >= 0 controls blur strength.
    scale = 1.0 / (1.0 + self.beta)  # <1.0
    """

    def __init__(self, beta=0.0, mode='bicubic'):
        self.beta = beta
        self.mode = mode

    def __call__(self, I):

        if self.beta <= 0:
            return I

        # shape handling
        squeeze = False
        if I.dim() == 2:
            H, W = I.shape
            I = I.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            squeeze = True
        elif I.dim() == 3:
            C, H, W = I.shape
            I = I.unsqueeze(0)               # [1,C,H,W]
        else:
            raise ValueError("I must be 2D or 3D tensor")

        scale = 1.0 / (1.0 + self.beta)  # <1.0
        H_down = max(1, int(H * scale))
        W_down = max(1, int(W * scale))

        # downsample: area mode → averaging
        I_down = F.interpolate(I, size=(H_down, W_down), mode='area')

        # upsample back to original: bicubic / bilinear
        I_blur = F.interpolate(I_down, size=(H, W), mode=self.mode, align_corners=False)

        if squeeze:
            return I_blur.squeeze(0).squeeze(0)
        else:
            return I_blur.squeeze(0)
        
class blur_beta_resample_2:
    """
    Edge-blur via upsample + downsample.
    beta >= 0 controls blur strength.
    up_factor = 1.0 + beta  # >1.0
    """

    def __init__(self, beta=0.0, mode='bicubic'):
        self.beta = beta
        self.mode = mode

    def __call__(self, I):

        if self.beta <= 0:
            return I

        # -------- shape handling --------
        squeeze = False
        if I.dim() == 2:
            H, W = I.shape
            I = I.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            squeeze = True
        elif I.dim() == 3:
            C, H, W = I.shape
            I = I.unsqueeze(0)               # [1,C,H,W]
        else:
            raise ValueError("I must be 2D or 3D tensor")

        # -------- upsample --------
        factor = 1.0 + self.beta
        H_up = max(1, int(H * factor))
        W_up = max(1, int(W * factor))

        I_up = F.interpolate(
            I,
            size=(H_up, W_up),
            mode='nearest',
            align_corners=None
        )

        # -------- downsample (blur happens here) --------
        I_blur = F.interpolate(I_up, size=(H, W), mode='area')

        # -------- restore shape --------
        if squeeze:
            return I_blur.squeeze(0).squeeze(0)
        else:
            return I_blur.squeeze(0)
        

class GaussianBlur:
    """
    Gaussian blur for 2D/3D tensor image.

    Args:
        sigma: std of Gaussian
        kernel_size: optional, auto-computed if None
    """

    def __init__(self, sigma=1.0, kernel_size=None):
        self.sigma = sigma

        if kernel_size is None:
            # 常用经验公式
            kernel_size = int(2 * round(3 * sigma) + 1)

        self.kernel_size = kernel_size
        self.kernel_1d = self._create_kernel()

    def _create_kernel(self):
        k = self.kernel_size
        sigma = self.sigma

        ax = torch.arange(k) - k // 2
        kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel /= kernel.sum()

        return kernel

    def __call__(self, I):

        if self.sigma <= 0:
            return I

        # ---------- shape handling ----------
        squeeze = False

        if I.dim() == 2:
            H, W = I.shape
            I = I.unsqueeze(0).unsqueeze(0)
            squeeze = True

        elif I.dim() == 3:
            C, H, W = I.shape
            I = I.unsqueeze(0)

        else:
            raise ValueError("I must be 2D or 3D tensor")

        C = I.shape[1]
        k = self.kernel_size

        # ---------- build separable kernel ----------
        k1d = self.kernel_1d.to(I.device, I.dtype)

        kernel_x = k1d.view(1,1,1,k).repeat(C,1,1,1)
        kernel_y = k1d.view(1,1,k,1).repeat(C,1,1,1)

        # ---------- apply blur ----------
        padding = k // 2

        I = F.conv2d(I, kernel_x, padding=(0,padding), groups=C)
        I = F.conv2d(I, kernel_y, padding=(padding,0), groups=C)

        # ---------- restore shape ----------
        if squeeze:
            return I.squeeze(0).squeeze(0)
        else:
            return I.squeeze(0)
        
class random_angle_composition():
    def __init__(self):
        pass

    def __call__(self, I):
        angle = np.random.uniform(0, 360)
        I_rot = transforms.functional.rotate(I, angle, resample=False, expand=False, center=None, fill=0)
        return I_rot