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